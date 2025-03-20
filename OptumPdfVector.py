#Loading Data with Databricks Auto Loader

spark.sql(f"USE CATALOG `{catalog_name}`")

df = (spark.readStream
      .format('cloudFiles')
      .option('cloudFiles.format', 'BINARYFILE')
      .load('dbfs:'+data_directory_path))
	  
	  
#Saving Raw Data as Delta Table

(df.writeStream
 .trigger(availableNow=True)
 .option("checkpointLocation", f'dbfs:{data_directory_path}/checkpoints/raw_data')
 .table(f'{schema_name}.pdf_raw').awaitTermination())
 
 
#  Data Processing and Chunk Parsing

import pyspark.sql.functions as F
# Import core components of Llama Index for text and token processing
from llama_index.core.node_parser import SimpleNodeParser
# Import core components of Llama Index for text and token processing
from llama_index.core.readers.base import Document
import io
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

parser = SimpleNodeParser()  # SimpleNodeParser for chunking

# Define UDF to extract text from PDF bytes with error handling
@F.udf("string")
def extract_pdf_text(content):
    try:
        pdf = io.BytesIO(content)  # Create file-like object from bytes
        reader = PdfReader(pdf)  # Read PDF
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
        return text
    except PdfReadError as e:
        return f"Error extracting PDF text: {e}"
    except Exception as e:
        return "Error: Unsupported or corrupted PDF"

# Define UDF for chunking using LlamaIndex
@F.udf("array<string>")
def llama_index_chunk_udf(content):
    try:
        document = Document(text=content)  # Wrap content in a Document object
        nodes = parser.get_nodes_from_documents([document])  # Parse content into nodes
        return [node.get_text() for node in nodes]
    except Exception as e:
        return [f"Error chunking content: {e}"]
		
		

# Steps to deploy

# Define UDF for Azure OpenAI embeddings
@F.udf("array<float>")
def azure_openai_embed_udf(content):
    try:
        if not content or not content.strip():
            raise ValueError("Empty or invalid content for embedding")

        import mlflow.deployments
        deploy_client = mlflow.deployments.get_deploy_client("databricks")
        response = deploy_client.predict(endpoint="embedding_aifoundry", inputs={"input": content})

        return response.data[0]['embedding']
    except Exception as e:
        return [f"Error chunking content: {e}"]
    
# Streaming pipeline
(
    spark.readStream.table(f'{schema_name}.pdf_raw')
        .withColumn("decoded_text", extract_pdf_text(F.col("content")))  # Extract text from PDF bytes
        .withColumn("chunks", F.explode(llama_index_chunk_udf(F.col("decoded_text"))))  # Apply LlamaIndex chunking
        .withColumn("embedding", azure_openai_embed_udf(F.col("chunks")))  # Apply embedding
        .selectExpr("path as url", "chunks as content", "embedding")  # Select final columns
        .writeStream
        .trigger(availableNow=True)
        .option("checkpointLocation", f'dbfs:{data_directory_path}/checkpoints/pdf_cleans')
        .table(f'{schema_name}.pdf_clean_embedding')
        .awaitTermination()
)



# Vector search implementation

# Import Databricks VectorSearch client for vector-based search operations
from databricks.vector_search.client import VectorSearchClient
# Import Databricks VectorSearch client for vector-based search operations
vsc = VectorSearchClient(disable_notice=True)
vector_name = "healthcare_medium"

if len(vsc.list_endpoints()) == 0 or vector_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    vsc.create_endpoint(name=vector_name, endpoint_type="STANDARD")

clean_table = f"{catalog_name}.{schema_name}.pdf_clean_embedding"
index_name = f"{catalog_name}.{schema_name}.healthcare_index"

vsc.create_delta_sync_index(
    endpoint_name=vector_name,
    index_name=index_name,
    source_table_name=clean_table,
    pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=3072,  # Match your model embedding size
    embedding_vector_column="embedding"
)


#Test


import mlflow.deployments

question = "How does Australia's healthcare system balance universal access through Medicare with the role of private supplementary insurance, and what benefits do Australians gain from each?"
deploy_client = mlflow.deployments.get_deploy_client("databricks")
response = deploy_client.predict(
    endpoint="embedding_aifoundry", inputs={"input": question})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(vector_name, index_name).similarity_search(
    query_vector=embeddings[0],
    columns=["url", "content"],
    num_results=1)
    
docs = results.get('result', {}).get('data_array', [])
print(docs)