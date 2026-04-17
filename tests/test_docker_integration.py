import logging
import pathlib
import socket
import subprocess
import time
import unittest

import grpc
from google.protobuf.json_format import MessageToJson
from ods_exd_api_box import exd_api, exd_grpc, ods


class TestDockerContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Docker-Image bauen
        subprocess.run(["docker", "build", "-t", "asam-ods-exd-api-microunidaq", "."], check=True)

        # Remove any stale container with the same name
        subprocess.run(
            ["docker", "rm", "-f", "test_container"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        example_file_path = pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "..", "data")
        data_folder = pathlib.Path(example_file_path).absolute().resolve()
        cp = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                "test_container",
                "-p",
                "50051:50051",
                "-v",
                f"{data_folder}:/data",
                "asam-ods-exd-api-microunidaq",
            ],
            stdout=subprocess.PIPE,
            check=True,
        )
        cls.container_id = cp.stdout.decode().strip()
        cls.__wait_for_port_ready()
        cls.__wait_for_grpc_ready()

    @classmethod
    def tearDownClass(cls):
        # Container stoppen
        subprocess.run(["docker", "stop", "test_container"], check=True)

    @classmethod
    def __wait_for_port_ready(cls, host="localhost", port=50051, timeout=30):
        start_time = time.time()
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return
            if time.time() - start_time > timeout:
                raise TimeoutError("Port did not become ready in time.")
            time.sleep(0.5)

    @classmethod
    def __wait_for_grpc_ready(cls, timeout=30):
        channel = grpc.insecure_channel("localhost:50051")
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        finally:
            channel.close()

    def test_container_health(self):
        channel = grpc.insecure_channel("localhost:50051")
        stub = exd_grpc.ExternalDataReaderStub(channel)
        self.assertIsNotNone(stub)
        grpc.channel_ready_future(channel).result(timeout=5)

    def test_structure(self):
        with grpc.insecure_channel("localhost:50051") as channel:
            service = exd_grpc.ExternalDataReaderStub(channel)

            handle = service.Open(exd_api.Identifier(url="/data/test5.hdf5", parameters=""), None)
            try:
                structure = service.GetStructure(exd_api.StructureRequest(handle=handle), None)
                logging.info(MessageToJson(structure))

                self.assertEqual(structure.name, "test5.hdf5")
                self.assertEqual(len(structure.groups), 4)
                for group in structure.groups:
                    self.assertEqual(group.number_of_rows, 114688)
                    self.assertEqual(len(group.channels), 2)
                self.assertEqual(structure.groups[0].id, 0)
                self.assertEqual(structure.groups[0].channels[0].id, 0)
                self.assertEqual(structure.groups[0].channels[1].id, 1)
            finally:
                service.Close(handle, None)

    def test_get_values(self):
        with grpc.insecure_channel("localhost:50051") as channel:
            service = exd_grpc.ExternalDataReaderStub(channel)

            handle = service.Open(exd_api.Identifier(url="/data/test5.hdf5", parameters=""), None)

            try:
                values = service.GetValues(
                    exd_api.ValuesRequest(handle=handle, group_id=0, channel_ids=[0, 1], start=0, limit=4),
                    None,
                )
                self.assertEqual(values.id, 0)
                self.assertEqual(len(values.channels), 2)
                self.assertEqual(values.channels[0].id, 0)
                self.assertEqual(values.channels[1].id, 1)
                logging.info(MessageToJson(values))

                # Channel 0: Time (DT_DOUBLE), computed from XInc=2e-05
                self.assertEqual(values.channels[0].values.data_type, ods.DataTypeEnum.DT_DOUBLE)
                time_values = list(values.channels[0].values.double_array.values)
                expected_time = [0.0, 2e-05, 4e-05, 6e-05]
                self.assertEqual(len(time_values), len(expected_time))
                for actual, expected in zip(time_values, expected_time):
                    self.assertAlmostEqual(actual, expected, places=15)
                # Channel 1: Data (DT_FLOAT)
                self.assertEqual(values.channels[1].values.data_type, ods.DataTypeEnum.DT_FLOAT)
                self.assertEqual(len(values.channels[1].values.float_array.values), 4)
            finally:
                service.Close(handle, None)
