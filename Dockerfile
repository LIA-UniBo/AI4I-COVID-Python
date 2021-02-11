FROM python:3

RUN pip install jupyter pandas sklearn matplotlib ipympl RISE jupyter_contrib_nbextensions sympy=1.71.1
RUN jupyter contrib nbextension install --system

# Copy raw data
COPY ./data /app/data

# During development, the notebooks folder will be overriden by a volume
COPY ./notebooks /app/notebooks

WORKDIR /app/notebooks

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]
