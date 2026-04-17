import logging
import pathlib
import unittest

from google.protobuf.json_format import MessageToJson
from ods_exd_api_box import ExternalDataReader, FileHandlerRegistry, exd_api, ods

from external_data_file import ExternalDataFile

# pylint: disable=E1101


class TestStringMethods(unittest.TestCase):
    log = logging.getLogger(__name__)

    def setUp(self):
        """Register ExternalDataFile handler before each test."""
        FileHandlerRegistry.register(file_type_name="test", factory=ExternalDataFile)

    def _get_example_file_path(self, file_name):
        example_file_path = pathlib.Path.joinpath(
            pathlib.Path(__file__).parent.resolve(), "..", "data", file_name
        )
        return pathlib.Path(example_file_path).absolute().as_uri()

    def test_open(self):
        service = ExternalDataReader()
        handle = service.Open(
            exd_api.Identifier(
                url=self._get_example_file_path("test5.hdf5"), parameters=""
            ),
            None,
        )
        try:
            pass
        finally:
            service.Close(handle, None)

    def test_structure(self):
        service = ExternalDataReader()
        handle = service.Open(
            exd_api.Identifier(
                url=self._get_example_file_path("test5.hdf5"), parameters=""
            ),
            None,
        )
        try:
            structure = service.GetStructure(
                exd_api.StructureRequest(handle=handle), None
            )
            self.assertEqual(structure.name, "test5.hdf5")
            self.assertEqual(len(structure.groups), 4)
            for group in structure.groups:
                self.assertEqual(group.number_of_rows, 114688)
                self.assertEqual(len(group.channels), 2)
            self.assertEqual(structure.groups[0].id, 0)
            self.assertEqual(structure.groups[0].name, "SinewaveGenerator")
            self.assertEqual(structure.groups[1].name, "Idle")
            self.assertEqual(structure.groups[2].name, "EncoderSpeed")
            self.assertEqual(structure.groups[3].name, "EncoderAngle")
            self.log.info(MessageToJson(structure))
            self.assertEqual(structure.groups[0].channels[0].id, 0)
            self.assertEqual(structure.groups[0].channels[1].id, 1)
        finally:
            service.Close(handle, None)

    def test_get_values(self):
        service = ExternalDataReader()
        handle = service.Open(
            exd_api.Identifier(
                url=self._get_example_file_path("test5.hdf5"), parameters=""
            ),
            None,
        )
        try:
            values = service.GetValues(
                exd_api.ValuesRequest(
                    handle=handle, group_id=0, channel_ids=[0, 1], start=0, limit=4
                ),
                None,
            )
            self.assertEqual(values.id, 0)
            self.assertEqual(len(values.channels), 2)
            self.assertEqual(values.channels[0].id, 0)
            self.assertEqual(values.channels[1].id, 1)
            self.log.info(MessageToJson(values))

            # Channel 0: Time (DT_DOUBLE), computed from XInc=2e-05
            self.assertEqual(
                values.channels[0].values.data_type, ods.DataTypeEnum.DT_DOUBLE
            )
            time_values = list(values.channels[0].values.double_array.values)
            expected_time = [0.0, 2e-05, 4e-05, 6e-05]
            self.assertEqual(len(time_values), len(expected_time))
            for actual, expected in zip(time_values, expected_time):
                self.assertAlmostEqual(actual, expected, places=15)
            # Channel 1: Data (DT_FLOAT)
            self.assertEqual(
                values.channels[1].values.data_type, ods.DataTypeEnum.DT_FLOAT
            )
            self.assertEqual(len(values.channels[1].values.float_array.values), 4)

        finally:
            service.Close(handle, None)
