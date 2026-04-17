import os
import tempfile
import unittest
from pathlib import Path

import grpc
import h5py
import numpy as np
from google.protobuf.json_format import MessageToJson
from ods_exd_api_box import ExternalDataReader, FileHandlerRegistry, exd_api

from external_data_file import ExternalDataFile
from tests.mock_servicer_context import MockServicerContext

# pylint: disable=E1101


def _make_hdf5(
    path: str,
    channels: list[dict] | None = None,
    measurement: dict | None = None,
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
    if measurement is None:
        measurement = {
            "DAQ Device Model": "TestModel",
            "DAQ Device Serial": "SN001",
            "DateTime": "2024-01-01T00:00:00+00:00",
            "Operator": "TestOperator",
        }

    with h5py.File(path, "w") as f:
        ft_group = f.create_group("FileType")
        ft_group.create_dataset("DsigntH5FileType", data=file_type.encode("utf-8"))
        meas_group = f.create_group("Measurement")
        for key, value in measurement.items():
            if isinstance(value, str):
                meas_group.create_dataset(key, data=value.encode("utf-8"))
            elif isinstance(value, float):
                meas_group.create_dataset(key, data=np.float32(value))
            else:
                meas_group.create_dataset(key, data=value)
        wf_group = f.create_group("Waveforms")
        for i, ch in enumerate(channels):
            samples = ch["samples"]
            chan_group = wf_group.create_group(f"Channel {i}")
            chan_group.attrs["Label"] = ch.get("label", f"Channel {i}")
            chan_group.attrs["NumPoints"] = np.int64(len(samples))
            chan_group.attrs["XInc"] = np.float64(ch.get("xinc", 0.001))
            chan_group.attrs["XUnits"] = ch.get("xunits", "s")
            chan_group.attrs["YUnits"] = ch.get("yunits", "V")
            if "input" in ch:
                chan_group.attrs["Input"] = ch["input"]
            if "xorg" in ch:
                chan_group.attrs["XOrg"] = ch["xorg"]
            for attr_key, attr_val in ch.get("extra_attrs", {}).items():
                chan_group.attrs[attr_key] = attr_val
            wf_group.create_dataset(f"Channel {i} Data", data=samples)


class TestExternalDataReader(unittest.TestCase):
    def setUp(self):
        FileHandlerRegistry.register(file_type_name="test", factory=ExternalDataFile)
        self.service = ExternalDataReader()
        self.mock_context = MockServicerContext()

    def test_open_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.hdf5")
            _make_hdf5(file_path)

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            try:
                handle = self.service.Open(identifier, None)
                self.assertIsNotNone(handle.uuid)
            finally:
                self.service.Close(handle, None)

    def test_open_non_existing_file(self):
        identifier = exd_api.Identifier(
            url="file:///non_existing_file.hdf5", parameters=""
        )
        with self.assertRaises(grpc.RpcError) as _:
            self.service.Open(identifier, self.mock_context)

        self.assertEqual(self.mock_context.code(), grpc.StatusCode.NOT_FOUND)

    def test_simple_example(self):
        main_file_path = Path.joinpath(
            Path(__file__).parent.resolve(), "..", "data", "test5.hdf5"
        )
        main_file_url = Path(main_file_path).absolute().resolve().as_uri()

        main_external_data_reader = ExternalDataReader()
        main_exd_api_handle = main_external_data_reader.Open(
            exd_api.Identifier(url=main_file_url, parameters=""), None
        )
        try:
            main_exd_api_structure = main_external_data_reader.GetStructure(
                exd_api.StructureRequest(handle=main_exd_api_handle), None
            )
            print(MessageToJson(main_exd_api_structure))

            print(
                MessageToJson(
                    main_external_data_reader.GetValues(
                        exd_api.ValuesRequest(
                            handle=main_exd_api_handle,
                            group_id=0,
                            channel_ids=[0, 1],
                            start=0,
                            limit=10,
                        ),
                        None,
                    )
                )
            )
        finally:
            main_external_data_reader.Close(main_exd_api_handle, None)

    def test_close_hdf5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.hdf5")
            _make_hdf5(file_path)

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            handle1 = self.service.Open(identifier, None)
            self.assertIsNotNone(handle1.uuid)

            handle2 = self.service.Open(identifier, None)
            self.assertIsNotNone(handle2.uuid)

            self.assertNotEqual(handle1.uuid, handle2.uuid)

            self.service.Close(handle1, None)
            self.service.Close(handle2, None)

    def test_hdf5_file_meta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "meta.hdf5")
            _make_hdf5(
                file_path,
                measurement={
                    "DAQ Device Model": "microUniDAQ-IEPE",
                    "DAQ Device Serial": "C5110030",
                    "DAQ Device Calibration Date": "20230605",
                    "DateTime": "2024-01-15T10:30:00+01:00",
                    "Operator": "TestUser",
                },
            )

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            handle = self.service.Open(identifier, None)
            try:
                file_content = self.service.GetStructure(
                    exd_api.StructureRequest(handle=handle), None
                )
                file_attributes = file_content.attributes.variables

                attribute = file_attributes.get("DAQ Device Model")
                self.assertIsNotNone(attribute)
                self.assertEqual(attribute.string_array.values[0], "microUniDAQ-IEPE")

                attribute = file_attributes.get("DAQ Device Serial")
                self.assertIsNotNone(attribute)
                self.assertEqual(attribute.string_array.values[0], "C5110030")

                attribute = file_attributes.get("DAQ Device Calibration Date")
                self.assertIsNotNone(attribute)
                self.assertEqual(attribute.string_array.values[0], "20230605")

                attribute = file_attributes.get("DateTime")
                self.assertIsNotNone(attribute)
                self.assertEqual(
                    attribute.string_array.values[0], "2024-01-15T10:30:00+01:00"
                )

                attribute = file_attributes.get("Operator")
                self.assertIsNotNone(attribute)
                self.assertEqual(attribute.string_array.values[0], "TestUser")

            finally:
                self.service.Close(handle, None)

    def test_not_my_file_wrong_type(self):
        """HDF5 with wrong file type marker should be rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "wrong.hdf5")
            _make_hdf5(file_path, file_type="SomeOtherFormat")

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            with self.assertRaises(Exception):
                self.service.Open(identifier, self.mock_context)

    def test_not_my_file_missing_waveforms(self):
        """HDF5 without Waveforms group should be rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "no_waveforms.hdf5")
            with h5py.File(file_path, "w") as f:
                ft = f.create_group("FileType")
                ft.create_dataset("DsigntH5FileType", data=b"DSignT Waveform")
                f.create_group("Measurement")

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            with self.assertRaises(Exception):
                self.service.Open(identifier, self.mock_context)

    def test_not_my_file_missing_filetype(self):
        """HDF5 without FileType group should be rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "no_filetype.hdf5")
            with h5py.File(file_path, "w") as f:
                f.create_group("Measurement")
                f.create_group("Waveforms")

            identifier = exd_api.Identifier(
                url=Path(file_path).resolve().as_uri(), parameters=""
            )
            with self.assertRaises(Exception):
                self.service.Open(identifier, self.mock_context)
