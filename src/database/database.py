from pymilvus import MilvusClient, DataType
from configs.config import Config

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri=Config.MILVUS_URI)

    def setup_collection(self):
        """Xóa cũ, tạo mới Schema và Collection"""
        if self.client.has_collection(Config.COLLECTION_NAME):
            self.client.drop_collection(Config.COLLECTION_NAME)

        schema = self.client.create_schema(auto_id=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="video_vector", datatype=DataType.FLOAT_VECTOR, dim=384)
        schema.add_field(field_name="caption",   datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="timestamp",  datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="video_id",   datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="camera_id",  datatype=DataType.VARCHAR, max_length=100)

        self.client.create_collection(
            collection_name=Config.COLLECTION_NAME, 
            schema=schema
        )
        
        # Tạo Index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="video_vector",
            metric_type="L2",
            index_type="IVF_FLAT",
            params={"nlist": 128}
        )
        self.client.create_index(Config.COLLECTION_NAME, index_params)
        self.client.load_collection(Config.COLLECTION_NAME)

    def insert_data(self, data):
        self.client.insert(collection_name=Config.COLLECTION_NAME, data=data)

    def search(self, query_vector, limit=2):
        results = self.client.search(
            collection_name=Config.COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            output_fields=["caption", "timestamp", "video_id", "camera_id"]
        )
        return results
