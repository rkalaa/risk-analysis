import subprocess
import os
from typing import List, Optional
from pyarrow import hdfs

class HadoopConnector:
    def __init__(self, hdfs_host: str = 'localhost', hdfs_port: int = 9000, hdfs_user: Optional[str] = None):
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        self.hdfs_user = hdfs_user or os.getenv('HDFS_USER', 'hadoop')
        self.client = hdfs.connect(host=self.hdfs_host, port=self.hdfs_port, user=self.hdfs_user)

    def list_directory(self, hdfs_path: str) -> List[str]:
        """
        List contents of an HDFS directory.
        
        :param hdfs_path: HDFS path to list
        :return: List of file and directory names
        """
        return self.client.ls(hdfs_path)

    def read_file(self, hdfs_path: str) -> bytes:
        """
        Read contents of an HDFS file.
        
        :param hdfs_path: HDFS path of the file to read
        :return: File contents as bytes
        """
        with self.client.open(hdfs_path, 'rb') as file:
            return file.read()

    def write_file(self, hdfs_path: str, content: bytes) -> None:
        """
        Write content to an HDFS file.
        
        :param hdfs_path: HDFS path of the file to write
        :param content: Content to write (as bytes)
        """
        with self.client.open(hdfs_path, 'wb') as file:
            file.write(content)

    def delete_file(self, hdfs_path: str) -> None:
        """
        Delete a file or directory from HDFS.
        
        :param hdfs_path: HDFS path to delete
        """
        self.client.delete(hdfs_path, recursive=True)

    def copy_from_local(self, local_path: str, hdfs_path: str) -> None:
        """
        Copy a file from the local filesystem to HDFS.
        
        :param local_path: Path of the local file
        :param hdfs_path: HDFS path to copy the file to
        """
        self.client.upload(hdfs_path, local_path)

    def copy_to_local(self, hdfs_path: str, local_path: str) -> None:
        """
        Copy a file from HDFS to the local filesystem.
        
        :param hdfs_path: HDFS path of the file to copy
        :param local_path: Local path to copy the file to
        """
        self.client.download(hdfs_path, local_path)

    def run_mapreduce_job(self, jar_path: str, main_class: str, input_path: str, output_path: str) -> None:
        """
        Run a MapReduce job on Hadoop.
        
        :param jar_path: Path to the JAR file containing the MapReduce job
        :param main_class: Main class of the MapReduce job
        :param input_path: HDFS input path
        :param output_path: HDFS output path
        """
        hadoop_cmd = f"hadoop jar {jar_path} {main_class} {input_path} {output_path}"
        subprocess.run(hadoop_cmd, shell=True, check=True)

    def get_file_info(self, hdfs_path: str) -> dict:
        """
        Get information about an HDFS file or directory.
        
        :param hdfs_path: HDFS path
        :return: Dictionary containing file information
        """
        return self.client.info(hdfs_path)

    def mkdir(self, hdfs_path: str) -> None:
        """
        Create a directory in HDFS.
        
        :param hdfs_path: HDFS path of the directory to create
        """
        self.client.mkdir(hdfs_path)

    def rename(self, hdfs_src_path: str, hdfs_dst_path: str) -> None:
        """
        Rename or move a file or directory in HDFS.
        
        :param hdfs_src_path: Source HDFS path
        :param hdfs_dst_path: Destination HDFS path
        """
        self.client.rename(hdfs_src_path, hdfs_dst_path)
