necessário ter o python 3.10 instalado na maquina


configurar ambiente:
python -m venv env
source env/bin/activate  # ou env\Scripts\activate no Windows
pip install -r requirements.txt



iniciar a aplicação : uvicorn main:app --reload


documentação da API (swagger): http://127.0.0.1:8000/docs
