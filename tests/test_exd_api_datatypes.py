import logging
import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from google.protobuf.json_format import MessageToJson
from ods_exd_api_box import ExternalDataReader, FileHandlerRegistry, exd_api, ods

from external_data_file import ExternalDataFile

# pylint: disable=E1101


def _make_hdf5(
    path: str,
    channels: list[dict] | None = None,
    file_type: str = "DSignT Waveform",
) -> None:
    """Create a minimal valid microUniDAQ HDF5 file."""
    if channels is None:
        channels = [
            {
                "label": "TestChannel",
                "samples": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "xinc": 0.001,
                "xunits": "s",
                "yunits": "V",
            }
        ]

    with h5py.File(path, "w") as f:
        ft_group = f.create_group("FileType")
        ft_group.create_dataset("DsigntH5FileType", data=file_type.encode("utf-8"))
        f.create_group("Measurement")
        wf_group = f.create_group("Waveforms")
        for i, ch in enumerate(channels):
            samples = ch["samples"]
            chan_group = wf_group.create_group(f"Channel {i}")
            chan_group.attrs["Label"] = ch.get("label", f"Channel {i}")
            chan_group.attrs["NumPoints"] = np.int64(len(samples))
            chan_group.attrs["XInc"] = np.float64(ch.get("xinc", 0.001))
            chan_group.attrs["XUnits"] = ch.get("xunits", "s")
            chan_group.attrs["YUnits"] = ch.get("yunits", "V")
            for attr_key, attr_val in ch.get("extra_attrs", {}).items():
                chan_group.attrs[attr_key] = attr_val
            wf_group.create_dataset(f"Channel {i} Data", data=samples)


class TestDataTypes(unittest.TestCase):
    log = logging.getLogger(__name__)

    def setUp(self):
        """Register ExternalDataFile handler before each test."""
        FileHandlerRegistry.register(file_type_name="test", factory=ExternalDataFile)

    def test_time_channel_is_double(self):
        """Channel 0 (time) should be DT_DOUBLE and marked independent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.hdf5")
            _make_hdf5(file_path)

            service = ExternalDataReader()
            handle = service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)
            try:
                structure = service.GetStructure(exd_api.StructureRequest(handle=handle), None)
                self.log.info(MessageToJson(structure))

                time_ch = structure.groups[0].channels[0]
                self.assertEqual(time_ch.name, "Time")
                self.assertEqual(time_ch.data_type, ods.DataTypeEnum.DT_DOUBLE)
                self.assertTrue("independent" in time_ch.attributes.variables)
                self.assertEqual(1, time_ch.attributes.variables["independent"].long_array.values[0])
            finally:
                service.Close(handle, None)

    def test_samples_channel_is_float(self):
        """Channel 1 (data samples) should be DT_FLOAT."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.hdf5")
            _make_hdf5(file_path)

            service = ExternalDataReader()
            handle = service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)
            try:
                structure = service.GetStructure(exd_api.StructureRequest(handle=handle), None)

                data_ch = structure.groups[0].channels[1]
                self.assertEqual(data_ch.name, "TestChannel")
                self.assertEqual(data_ch.data_type, ods.DataTypeEnum.DT_FLOAT)
                self.assertTrue("independent" not in data_ch.attributes.variables)
            finally:
                service.Close(handle, None)

    def test_time_values_from_xinc(self):
        """Time channel values should be computed from XInc attribute."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.hdf5")
            _make_hdf5(
                file_path,
                channels=[
                    {
                        "label": "TestCh",
                        "samples": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
                        "xinc": 0.01,
                        "xunits": "s",
                        "yunits": "mV",
                    }
                ],
            )

            service = ExternalDataReader()
            handle = service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)
            try:
                values = service.GetValues(
                    exd_api.ValuesRequest(handle=handle, group_id=0, start=0, limit=5, channel_ids=[0]), None
                )
                self.assertEqual(values.channels[0].values.data_type, ods.DataTypeEnum.DT_DOUBLE)
                expected = [0.0, 0.01, 0.02, 0.03, 0.04]
                for i, (actual, exp) in enumerate(
                    zip(values.channels[0].values.double_array.values, expected)
                ):
                    self.assertAlmostEqual(float(actual), exp, places=10, msg=f"index {i}")
            finally:
                service.Close(handle, None)

    def test_invalid_file_rejected(self):
        """An HDF5 file with a wrong file type marker should be rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "bad.hdf5")
            _make_hdf5(file_path, file_type="NotMyFormat")

            service = ExternalDataReader()
            with self.assertRaises(Exception):
                service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)

    def test_missing_filetype_rejected(self):
        """An HDF5 file without FileType/DsigntH5FileType should be rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "notype.hdf5")
            with h5py.File(file_path, "w") as f:
                f.create_group("Measurement")
                wf = f.create_group("Waveforms")
                g = wf.create_group("Channel 0")
                g.attrs["Label"] = "Test"
                g.attrs["NumPoints"] = np.int64(3)
                g.attrs["XInc"] = np.float64(0.001)
                g.attrs["XUnits"] = "s"
                g.attrs["YUnits"] = "V"
                wf.create_dataset("Channel 0 Data", data=np.array([1, 2, 3], dtype=np.float32))

            service = ExternalDataReader()
            with self.assertRaises(Exception):
                service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)

    def test_channel_units(self):
        """XUnits and YUnits should map to channel unit_string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "units.hdf5")
            _make_hdf5(
                file_path,
                channels=[{
                    "label": "Accel", "samples": np.array([1.0], dtype=np.float32),
                    "xinc": 0.001, "xunits": "ms", "yunits": "g",
                }],
            )

            service = ExternalDataReader()
            handle = service.Open(exd_api.Identifier(url=Path(file_path).resolve().as_uri(), parameters=""), None)
            try:
                structure = service.GetStructure(exd_api.StructureRequest(handle=handle), None)
                self.assertEqual(structure.groups[0].channels[0].unit_string, "ms")
                self.assertEqual(structure.groups[0].channels[1].unit_string, "g")
            finally:
                service.Close(handle, None)
