FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Create conda env and activate it
RUN conda update -n base -c defaults conda && \
    conda create -n stocksentiment python=3.9 && \
    conda install -n stocksentiment pip && \
    conda run -n stocksentiment pip install streamlit  panda matplotlib numpy seaborn altair tensorflow-cpu==2.16.1 joblib

# Copy the app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Use conda-run instead of conda activate
CMD ["conda", "run", "-n", "stocksentiment", "streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]