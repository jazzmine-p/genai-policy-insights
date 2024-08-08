FROM python:3.11

WORKDIR ./

COPY ./requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . /

# List the contents of the /code directory to verify files are copied correctly
RUN ls -R ./

# Change permissions to allow writing to the directory
RUN chmod -R 777 ./

# Create a logs directory and set permissions
RUN mkdir ./logs && chmod 777 ./logs

# Create a cache directory within the application's working directory
RUN mkdir /.cache && chmod -R 777 /.cache

WORKDIR /code/code

RUN --mount=type=secret,id=LLAMA_API_KEY,mode=0444,required=true 
RUN --mount=type=secret,id=HUGGING_FACE_TOKEN,mode=0444,required=true 
RUN --mount=type=secret,id=HUGGING_FACE_WRITE_TOKEN,mode=0444,required=true 
RUN --mount=type=secret,id=LITERAL_API_KEY,mode=0444,required=true 
RUN --mount=type=secret,id=CHAINLIT_AUTH_SECRET,mode=0444,required=true 
RUN --mount=type=secret,id=OPENAI_API_KEY,mode=0444,required=true 

# Default command to run the application
CMD ["sh", "-c", "python -m modules.vectorstore.store_manager && chainlit run main.py --host 0.0.0.0 --port 7860"]