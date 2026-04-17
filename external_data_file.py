"""ASAM ODS EXD API implementation for microUniDAQ HDF5 files."""

from __future__ import annotations

import re
from typing import Any, override

import h5py
import numpy as np
from ods_exd_api_box import ExdFileInterface, NotMyFileError, exd_api, ods

FILE_TYPE_KEY = "FileType/DsigntH5FileType"
EXPECTED_FILE_TYPE = "DSignT Waveform"


def _to_str(value: Any) -> str:
    """Normalize an HDF5 scalar (bytes, numpy scalar, str) to a Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _channel_sort_key(name: str) -> tuple[int, str]:
    """Sort key that orders 'Channel 2' before 'Channel 10' numerically."""
    m = re.search(r"(\d+)$", name)
    if m:
        return (int(m.group(1)), name)
    return (0, name)


class ExternalDataFile(ExdFileInterface):
    """Class for handling microUniDAQ HDF5 files."""

    @classmethod
    @override
    def create(cls, file_path: str, parameters: str) -> ExdFileInterface:
        """Factory method to create a file handler instance."""
        return cls(file_path, parameters)

    @override
    def __init__(self, file_path: str, parameters: str = ""):
        self.file_path = file_path
        self.parameters = parameters
        self.hdf5 = h5py.File(file_path, "r")
        self._validate_structure()
        self._channel_pairs = self._get_channel_pairs()

    def _validate_structure(self) -> None:
        """Validate that the HDF5 file is a microUniDAQ waveform file.

        Raises ValueError (mapped to 'not my file') when the content does not
        match the expected layout.
        """
        if FILE_TYPE_KEY not in self.hdf5:
            self.hdf5.close()
            raise NotMyFileError(f"Not a microUniDAQ file: missing '{FILE_TYPE_KEY}'")

        file_type_value = _to_str(self.hdf5[FILE_TYPE_KEY][()])
        if file_type_value != EXPECTED_FILE_TYPE:
            self.hdf5.close()
            raise NotMyFileError(f"Not a microUniDAQ file: expected '{EXPECTED_FILE_TYPE}', got '{file_type_value}'")

        if "Measurement" not in self.hdf5:
            self.hdf5.close()
            raise NotMyFileError("Not a microUniDAQ file: missing 'Measurement' group")

        if "Waveforms" not in self.hdf5:
            self.hdf5.close()
            raise NotMyFileError("Not a microUniDAQ file: missing 'Waveforms' group")

    def _get_channel_pairs(self) -> list[tuple[str, str]]:
        """Return sorted (group_name, dataset_name) pairs from 'Waveforms/'.

        Pairs are sorted by the numeric suffix of the group name for stable
        group-ID assignment across reopens.

        Raises ValueError when a channel group has no matching data dataset.
        """
        waveforms = self.hdf5["Waveforms"]
        group_names = [name for name in waveforms.keys() if isinstance(waveforms[name], h5py.Group)]
        group_names.sort(key=_channel_sort_key)

        pairs: list[tuple[str, str]] = []
        for gname in group_names:
            dname = f"{gname} Data"
            if dname not in waveforms or not isinstance(waveforms[dname], h5py.Dataset):
                self.hdf5.close()
                raise ValueError(f"Not a microUniDAQ file: missing dataset '{dname}' for group '{gname}'")
            pairs.append((gname, dname))

        if not pairs:
            self.hdf5.close()
            raise ValueError("Not a microUniDAQ file: no channel groups found in 'Waveforms'")

        return pairs

    @override
    def close(self) -> None:
        """Close the external data file."""
        self.hdf5.close()

    @override
    def fill_structure(self, structure: exd_api.StructureResult) -> None:
        """Fill the structure of the external data file."""
        hdf5 = self.hdf5

        # File-level attributes from Measurement group
        measurement = hdf5["Measurement"]
        for key in sorted(measurement.keys()):
            ds = measurement[key]
            if isinstance(ds, h5py.Dataset):
                structure.attributes.variables[key].string_array.values.append(_to_str(ds[()]))

        # Resolve labels (for group and channel names) with uniqueness check
        labels: list[str] = []
        for gname, _ in self._channel_pairs:
            group = hdf5[f"Waveforms/{gname}"]
            label = _to_str(group.attrs["Label"]) if "Label" in group.attrs else gname
            labels.append(label)

        # Deduplicate: if any labels collide, append _<index> to all instances of that label
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        resolved_labels: list[str] = []
        for i, label in enumerate(labels):
            if label_counts[label] > 1:
                resolved_labels.append(f"{label}_{i}")
            else:
                resolved_labels.append(label)

        # Groups
        for group_index, (gname, _dname) in enumerate(self._channel_pairs):
            chan_group = hdf5[f"Waveforms/{gname}"]
            attrs = chan_group.attrs
            num_points = int(attrs["NumPoints"])

            new_group = exd_api.StructureResult.Group()
            new_group.name = resolved_labels[group_index]
            new_group.id = group_index
            new_group.total_number_of_channels = 2
            new_group.number_of_rows = num_points

            # Store all HDF5 channel-group attributes as group attributes
            for attr_name in sorted(attrs.keys()):
                new_group.attributes.variables[attr_name].string_array.values.append(_to_str(attrs[attr_name]))

            # Channel 0: Time (independent)
            time_channel = exd_api.StructureResult.Channel()
            time_channel.name = "Time"
            time_channel.id = 0
            time_channel.data_type = ods.DataTypeEnum.DT_DOUBLE
            time_channel.unit_string = _to_str(attrs["XUnits"]) if "XUnits" in attrs else ""
            time_channel.attributes.variables["independent"].long_array.values.append(1)
            new_group.channels.append(time_channel)

            # Channel 1: Data samples
            data_channel = exd_api.StructureResult.Channel()
            data_channel.name = resolved_labels[group_index]
            data_channel.id = 1
            data_channel.data_type = ods.DataTypeEnum.DT_FLOAT
            data_channel.unit_string = _to_str(attrs["YUnits"]) if "YUnits" in attrs else ""
            new_group.channels.append(data_channel)

            structure.groups.append(new_group)

    @override
    def get_values(self, request: exd_api.ValuesRequest) -> exd_api.ValuesResult:
        """Get values from the external data file."""
        hdf5 = self.hdf5

        if request.group_id < 0 or request.group_id >= len(self._channel_pairs):
            raise ValueError(f"Invalid group id {request.group_id}!")

        gname, dname = self._channel_pairs[request.group_id]
        chan_group = hdf5[f"Waveforms/{gname}"]
        data_dataset = hdf5[f"Waveforms/{dname}"]
        num_points = int(chan_group.attrs["NumPoints"])

        if request.start > num_points:
            raise ValueError(f"Channel start index {request.start} out of range!")

        end_index = min(request.start + request.limit, num_points)
        xinc = float(chan_group.attrs["XInc"])

        rv = exd_api.ValuesResult(id=request.group_id)

        for channel_id in request.channel_ids:
            new_channel_values = exd_api.ValuesResult.ChannelValues()
            new_channel_values.id = channel_id

            if channel_id == 0:
                # Time channel: computed from XInc
                new_channel_values.values.data_type = ods.DataTypeEnum.DT_DOUBLE
                time_values = np.arange(request.start, end_index, dtype=np.float64) * xinc
                new_channel_values.values.double_array.values[:] = time_values
            elif channel_id == 1:
                # Data channel: read from dataset
                new_channel_values.values.data_type = ods.DataTypeEnum.DT_FLOAT
                samples = data_dataset[request.start : end_index]
                new_channel_values.values.float_array.values[:] = samples
            else:
                raise ValueError(f"Invalid channel id {channel_id}!")

            rv.channels.append(new_channel_values)

        return rv


if __name__ == "__main__":
    from ods_exd_api_box import serve_plugin

    serve_plugin(
        file_type_name="MICROUNIDAQ",
        file_type_factory=ExternalDataFile.create,
        file_type_file_patterns=["*.hdf5", "*.h5"],
    )
