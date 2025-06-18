# Use uma imagem oficial Python slim para evitar excesso de peso
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos necessários
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc

# Instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir tensorflow==2.13.0
RUN pip install --no-cache-dir -r requirements.txt


# Copia a aplicação
COPY . .

# Expõe a porta do uvicorn
EXPOSE 80

# Comando para iniciar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "app"]
