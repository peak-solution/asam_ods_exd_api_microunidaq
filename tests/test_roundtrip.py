"""Roundtrip tests: write microUniDAQ HDF5 files, read back via ExternalDataFile, verify.

Each test writes one or more HDF5 files into a TemporaryDirectory, opens them
through the ExdFileInterface / ExternalDataReader and checks that the returned
structure and values match exactly what was written.

Data-type mapping for microUniDAQ:
  Time channel (ch 0)  → DT_DOUBLE  (double_array) computed from XInc
  Data channel (ch 1)  → DT_FLOAT   (float_array)  read from dataset
"""

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from ods_exd_api_box import ExternalDataReader, FileHandlerRegistry, exd_api, ods

from external_data_file import ExternalDataFile

# pylint: disable=E1101


def _uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _make_hdf5(
    path: str,
    channels: list[dict] | None = None,
    measurement: dict | None = None,
    file_type: str = "DSignT Waveform",
) -> str:
    """Create a microUniDAQ HDF5 file and return its path."""
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
            "DateTime": "2024-01-01T00:00:00+00:00",
        }

    with h5py.File(path, "w") as f:
        ft = f.create_group("FileType")
        ft.create_dataset("DsigntH5FileType", data=file_type.encode("utf-8"))
        meas = f.create_group("Measurement")
        for key, value in measurement.items():
            if isinstance(value, str):
                meas.create_dataset(key, data=value.encode("utf-8"))
            elif isinstance(value, float):
                meas.create_dataset(key, data=np.float32(value))
            else:
                meas.create_dataset(key, data=value)
        wf = f.create_group("Waveforms")
        for i, ch in enumerate(channels):
            samples = ch["samples"]
            g = wf.create_group(f"Channel {i}")
            g.attrs["Label"] = ch.get("label", f"Channel {i}")
            g.attrs["NumPoints"] = np.int64(len(samples))
            g.attrs["XInc"] = np.float64(ch.get("xinc", 0.001))
            g.attrs["XUnits"] = ch.get("xunits", "s")
            g.attrs["YUnits"] = ch.get("yunits", "V")
            if "xorg" in ch:
                g.attrs["XOrg"] = ch["xorg"]
            if "input_name" in ch:
                g.attrs["Input"] = ch["input_name"]
            for attr_key, attr_val in ch.get("extra_attrs", {}).items():
                g.attrs[attr_key] = attr_val
            wf.create_dataset(f"Channel {i} Data", data=samples)

    return path


class TestRoundtrip(unittest.TestCase):
    """Write HDF5 via h5py, read back through ExternalDataFile, assert correctness."""

    def setUp(self):
        FileHandlerRegistry.register(file_type_name="test", factory=ExternalDataFile)
        self.service = ExternalDataReader()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.tmp_dir, name)

    def _open(self, path: str):
        return self.service.Open(exd_api.Identifier(url=_uri(path), parameters=""), None)

    def _structure(self, handle) -> exd_api.StructureResult:
        return self.service.GetStructure(exd_api.StructureRequest(handle=handle), None)

    def _values(
        self,
        handle,
        group_id: int,
        channel_ids: list,
        start: int = 0,
        limit: int = 999999,
    ) -> exd_api.ValuesResult:
        return self.service.GetValues(
            exd_api.ValuesRequest(
                handle=handle,
                group_id=group_id,
                channel_ids=channel_ids,
                start=start,
                limit=limit,
            ),
            None,
        )

    def _assert_floats(self, actual, expected, places: int = 5):
        self.assertEqual(len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(float(a), float(e), places=places, msg=f"index {i}")

    # ==================================================================
    # Basic data roundtrip
    # ==================================================================

    def test_float32_samples_roundtrip(self):
        """Written float32 samples should come back as DT_FLOAT."""
        vals = np.array([1.5, -2.25, 0.0], np.float32)
        path = _make_hdf5(
            self._path("f32.hdf5"),
            channels=[{"label": "ch_f32", "samples": vals, "yunits": "V"}],
        )
        handle = self._open(path)
        try:
            ch = self._structure(handle).groups[0].channels[1]
            self.assertEqual(ch.data_type, ods.DataTypeEnum.DT_FLOAT)
            result = self._values(handle, 0, [1])
            self._assert_floats(result.channels[0].values.float_array.values, vals.tolist())
        finally:
            self.service.Close(handle, None)

    def test_time_channel_is_double(self):
        """Time channel must be DT_DOUBLE with values computed from XInc."""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32)
        path = _make_hdf5(
            self._path("time.hdf5"),
            channels=[{"label": "ch", "samples": vals, "xinc": 0.01, "xunits": "s"}],
        )
        handle = self._open(path)
        try:
            ch = self._structure(handle).groups[0].channels[0]
            self.assertEqual(ch.name, "Time")
            self.assertEqual(ch.data_type, ods.DataTypeEnum.DT_DOUBLE)
            result = self._values(handle, 0, [0])
            expected = [i * 0.01 for i in range(5)]
            self._assert_floats(result.channels[0].values.double_array.values, expected, places=10)
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Label → group & channel name
    # ==================================================================

    def test_label_becomes_group_name(self):
        path = _make_hdf5(
            self._path("label.hdf5"),
            channels=[{"label": "Accelerometer", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            grp = self._structure(handle).groups[0]
            self.assertEqual(grp.name, "Accelerometer")
            self.assertEqual(grp.channels[1].name, "Accelerometer")
        finally:
            self.service.Close(handle, None)

    def test_duplicate_labels_get_suffix(self):
        """When two channels share the same Label, group names get _N suffix."""
        path = _make_hdf5(
            self._path("dup.hdf5"),
            channels=[
                {"label": "Voltage", "samples": np.array([1.0], np.float32)},
                {"label": "Voltage", "samples": np.array([2.0], np.float32)},
                {"label": "Current", "samples": np.array([3.0], np.float32)},
            ],
        )
        handle = self._open(path)
        try:
            structure = self._structure(handle)
            names = [g.name for g in structure.groups]
            self.assertEqual(names, ["Voltage_0", "Voltage_1", "Current"])
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Units
    # ==================================================================

    def test_xunits_and_yunits(self):
        path = _make_hdf5(
            self._path("units.hdf5"),
            channels=[
                {
                    "label": "ch",
                    "samples": np.array([1.0], np.float32),
                    "xunits": "ms",
                    "yunits": "mV",
                }
            ],
        )
        handle = self._open(path)
        try:
            grp = self._structure(handle).groups[0]
            self.assertEqual(grp.channels[0].unit_string, "ms")  # time
            self.assertEqual(grp.channels[1].unit_string, "mV")  # data
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Multiple groups
    # ==================================================================

    def test_multiple_groups_roundtrip(self):
        """Each channel pair becomes a separate group addressable by group_id."""
        path = _make_hdf5(
            self._path("multi.hdf5"),
            channels=[
                {
                    "label": "Ch_A",
                    "samples": np.array([10.0, 20.0], np.float32),
                    "yunits": "V",
                },
                {
                    "label": "Ch_B",
                    "samples": np.array([30.0, 40.0], np.float32),
                    "yunits": "A",
                },
            ],
        )
        handle = self._open(path)
        try:
            structure = self._structure(handle)
            self.assertEqual(len(structure.groups), 2)
            self.assertEqual(structure.groups[0].id, 0)
            self.assertEqual(structure.groups[1].id, 1)

            r0 = self._values(handle, 0, [1])
            self._assert_floats(r0.channels[0].values.float_array.values, [10.0, 20.0])
            r1 = self._values(handle, 1, [1])
            self._assert_floats(r1.channels[0].values.float_array.values, [30.0, 40.0])
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Reopen stability
    # ==================================================================

    def test_reopen_yields_identical_structure(self):
        """Opening the same file twice must produce identical group/channel IDs and names."""
        path = _make_hdf5(
            self._path("stable.hdf5"),
            channels=[
                {"label": "Alpha", "samples": np.array([1.0], np.float32)},
                {"label": "Beta", "samples": np.array([2.0], np.float32)},
                {"label": "Gamma", "samples": np.array([3.0], np.float32)},
            ],
        )
        handle1 = self._open(path)
        handle2 = self._open(path)
        try:
            s1 = self._structure(handle1)
            s2 = self._structure(handle2)
            self.assertEqual(len(s1.groups), len(s2.groups))
            for g1, g2 in zip(s1.groups, s2.groups):
                self.assertEqual(g1.id, g2.id)
                self.assertEqual(g1.name, g2.name)
                for c1, c2 in zip(g1.channels, g2.channels):
                    self.assertEqual(c1.id, c2.id)
                    self.assertEqual(c1.name, c2.name)
                    self.assertEqual(c1.data_type, c2.data_type)
        finally:
            self.service.Close(handle1, None)
            self.service.Close(handle2, None)

    # ==================================================================
    # Paging: start / limit
    # ==================================================================

    def test_paging_middle_slice(self):
        n = 50
        vals = np.arange(n, dtype=np.float32)
        path = _make_hdf5(
            self._path("paging.hdf5"),
            channels=[{"label": "ramp", "samples": vals, "xinc": 0.001}],
        )
        handle = self._open(path)
        try:
            result = self._values(handle, 0, [1], start=10, limit=5)
            self._assert_floats(result.channels[0].values.float_array.values, vals[10:15].tolist())
        finally:
            self.service.Close(handle, None)

    def test_paging_limit_clamps_at_end(self):
        vals = np.array([1.0, 2.0, 3.0], np.float32)
        path = _make_hdf5(self._path("clamp.hdf5"), channels=[{"label": "ch", "samples": vals}])
        handle = self._open(path)
        try:
            result = self._values(handle, 0, [1], start=0, limit=9999)
            self.assertEqual(len(result.channels[0].values.float_array.values), 3)
        finally:
            self.service.Close(handle, None)

    def test_paging_first_row_only(self):
        vals = np.array([10.0, 20.0, 30.0], np.float32)
        path = _make_hdf5(self._path("first.hdf5"), channels=[{"label": "ch", "samples": vals}])
        handle = self._open(path)
        try:
            result = self._values(handle, 0, [1], start=0, limit=1)
            self._assert_floats(result.channels[0].values.float_array.values, [10.0])
        finally:
            self.service.Close(handle, None)

    def test_paging_last_row(self):
        vals = np.array([10.0, 20.0, 30.0], np.float32)
        path = _make_hdf5(self._path("last.hdf5"), channels=[{"label": "ch", "samples": vals}])
        handle = self._open(path)
        try:
            result = self._values(handle, 0, [1], start=2, limit=1)
            self._assert_floats(result.channels[0].values.float_array.values, [30.0])
        finally:
            self.service.Close(handle, None)

    def test_paging_time_and_data_same_slice(self):
        """Requesting both channels with start/limit must return the same slice."""
        n = 20
        vals = np.arange(n, dtype=np.float32)
        path = _make_hdf5(
            self._path("slice.hdf5"),
            channels=[{"label": "ch", "samples": vals, "xinc": 0.1}],
        )
        handle = self._open(path)
        try:
            result = self._values(handle, 0, [0, 1], start=5, limit=4)
            # Time
            expected_time = [i * 0.1 for i in range(5, 9)]
            self._assert_floats(result.channels[0].values.double_array.values, expected_time)
            # Data
            self._assert_floats(result.channels[1].values.float_array.values, vals[5:9].tolist())
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Edge cases
    # ==================================================================

    def test_single_row_roundtrip(self):
        path = _make_hdf5(
            self._path("single.hdf5"),
            channels=[{"label": "ch", "samples": np.array([42.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            self.assertEqual(self._structure(handle).groups[0].number_of_rows, 1)
            result = self._values(handle, 0, [1])
            self._assert_floats(result.channels[0].values.float_array.values, [42.0])
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Channel group attributes
    # ==================================================================

    def test_channel_group_attributes_propagated(self):
        """All HDF5 attributes on a channel group should appear in group.attributes."""
        path = _make_hdf5(
            self._path("attrs.hdf5"),
            channels=[
                {
                    "label": "ch",
                    "samples": np.array([1.0], np.float32),
                    "xinc": 0.001,
                    "xunits": "s",
                    "yunits": "V",
                    "input_name": "AIN 1",
                    "extra_attrs": {"Coupling": "DC", "Range": "12V"},
                }
            ],
        )
        handle = self._open(path)
        try:
            grp = self._structure(handle).groups[0]
            attrs = grp.attributes.variables
            self.assertEqual(attrs["Label"].string_array.values[0], "ch")
            self.assertEqual(attrs["Input"].string_array.values[0], "AIN 1")
            self.assertEqual(attrs["Coupling"].string_array.values[0], "DC")
            self.assertEqual(attrs["Range"].string_array.values[0], "12V")
            self.assertEqual(attrs["XUnits"].string_array.values[0], "s")
            self.assertEqual(attrs["YUnits"].string_array.values[0], "V")
        finally:
            self.service.Close(handle, None)

    def test_first_channel_is_independent(self):
        """Channel 0 must carry independent=1; channel 1 must not."""
        path = _make_hdf5(
            self._path("indep.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            grp = self._structure(handle).groups[0]
            time_ch = grp.channels[0]
            self.assertIn("independent", time_ch.attributes.variables)
            self.assertEqual(time_ch.attributes.variables["independent"].long_array.values[0], 1)
            data_ch = grp.channels[1]
            self.assertNotIn("independent", data_ch.attributes.variables)
        finally:
            self.service.Close(handle, None)

    def test_total_number_of_channels(self):
        path = _make_hdf5(
            self._path("total.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            grp = self._structure(handle).groups[0]
            # Always 2: time + data
            self.assertEqual(grp.total_number_of_channels, 2)
            self.assertEqual(len(grp.channels), 2)
        finally:
            self.service.Close(handle, None)

    def test_structure_identifier_url_is_set(self):
        path = _make_hdf5(
            self._path("url.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            self.assertNotEqual(self._structure(handle).identifier.url, "")
        finally:
            self.service.Close(handle, None)

    def test_values_result_carries_group_id(self):
        """ValuesResult.id must equal the requested group_id."""
        path = _make_hdf5(
            self._path("grpid.hdf5"),
            channels=[{"label": f"ch{i}", "samples": np.array([float(i)], np.float32)} for i in range(3)],
        )
        handle = self._open(path)
        try:
            for gid in range(3):
                result = self._values(handle, gid, [1])
                self.assertEqual(result.id, gid)
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Measurement metadata
    # ==================================================================

    def test_measurement_metadata_in_file_attributes(self):
        path = _make_hdf5(
            self._path("meta.hdf5"),
            measurement={"DAQ Device Model": "myModel", "Operator": "Jane"},
        )
        handle = self._open(path)
        try:
            attrs = self._structure(handle).attributes.variables
            self.assertEqual(attrs["DAQ Device Model"].string_array.values[0], "myModel")
            self.assertEqual(attrs["Operator"].string_array.values[0], "Jane")
        finally:
            self.service.Close(handle, None)

    # ==================================================================
    # Error cases
    # ==================================================================

    def test_invalid_group_id_raises(self):
        path = _make_hdf5(
            self._path("err_grp.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            with self.assertRaises(ValueError):
                self._values(handle, 999, [0])
        finally:
            self.service.Close(handle, None)

    def test_invalid_channel_id_raises(self):
        path = _make_hdf5(
            self._path("err_ch.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            with self.assertRaises(ValueError):
                self._values(handle, 0, [999])
        finally:
            self.service.Close(handle, None)

    def test_start_beyond_row_count_raises(self):
        path = _make_hdf5(
            self._path("err_start.hdf5"),
            channels=[{"label": "ch", "samples": np.array([1.0, 2.0], np.float32)}],
        )
        handle = self._open(path)
        try:
            with self.assertRaises(ValueError):
                self._values(handle, 0, [1], start=999)
        finally:
            self.service.Close(handle, None)

    def test_not_my_file_rejected(self):
        """An HDF5 file with wrong file type should be rejected on open."""
        path = _make_hdf5(self._path("bad.hdf5"), file_type="NotMyFormat")
        with self.assertRaises(Exception):
            self._open(path)
