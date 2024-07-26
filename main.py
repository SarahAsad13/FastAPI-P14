from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from py2neo import Graph, Node
import spacy
import uuid
import pandas as pd
import io
from pdfminer.high_level import extract_text

# Initialize FastAPI app
app = FastAPI()

# Initialize Neo4j connection
uri = "bolt://127.0.0.1:7687"
username = "neo4j"  # Update with your Neo4j username
password = "hello123"  # Update with your Neo4j password
graph = Graph(uri, auth=(username, password))

# Load NLP models and components
nlp = spacy.load("en_core_web_sm")

# Allowing CORS Headers
origins = ["http://localhost:8000", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")

# Root endpoint
@app.get('/')
async def root():
    return {'message': 'This is the root of the app'}

# In-memory storage for resumes and parsed resumes
resumes_storage = {}
parsed_resumes_storage = {}

# Store the ID of the most recently uploaded resume
latest_resume_id = None

# Endpoint to upload and parse a resume
@app.post("/parse_resume/")
async def parse_resume(file: UploadFile = File(...)):
    global latest_resume_id
    try:
        # Generate a unique ID for the resume
        resume_id = str(uuid.uuid4())

        # Read the file content into memory
        content = await file.read()

        # Store the content in memory
        resumes_storage[resume_id] = content

        # Use pdfminer to extract text from the PDF
        pdf_file = io.BytesIO(content)
        parsed_content = extract_text(pdf_file)

        # Store the parsed content in memory
        parsed_resumes_storage[resume_id] = parsed_content

        # Update the latest resume ID
        latest_resume_id = resume_id

        return {"message": "Resume uploaded and parsed successfully.", "resume_id": resume_id, "parsed_resume": parsed_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to extract entities using NER and store in Neo4j
@app.post("/extract_entities/")
async def extract_entities():
    global latest_resume_id
    try:
        # Ensure there's a recent resume ID available
        if latest_resume_id is None:
            raise HTTPException(status_code=404, detail="No recent resume found")
        
        # Retrieve the parsed resume from memory
        if latest_resume_id not in parsed_resumes_storage:
            raise HTTPException(status_code=404, detail="Parsed resume not found")
        
        parsed_content = parsed_resumes_storage[latest_resume_id]

        # Use SpaCy to extract entities
        doc = nlp(parsed_content)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Delete existing entities from Neo4j
        graph.run("MATCH (e:Entity) DELETE e")

        # Store new entities in Neo4j
        for entity_text, entity_label in entities:
            node = Node("Entity", name=entity_text, label=entity_label)
            graph.merge(node, "Entity", "name")
        
        return {"message": "Entities extracted and stored successfully.", "entities": entities}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to download results as a CSV file
@app.get("/download_results/")
async def download_results():
    try:
        # Query Neo4j to retrieve entities
        query = "MATCH (e:Entity) RETURN e.name AS name, e.label AS label"
        result = graph.run(query).data()

        # Convert Neo4j result to pandas DataFrame
        df = pd.DataFrame(result)

        # Prepare data for download (e.g., to CSV)
        csv_data = df.to_csv(index=False)

        # Prepare HTTP response with CSV data for download
        response = Response(content=csv_data)
        response.headers["Content-Disposition"] = 'attachment; filename="neo4j_data.csv"'
        response.headers["Content-Type"] = "text/csv"
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
