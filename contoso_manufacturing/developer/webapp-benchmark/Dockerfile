FROM openvino/ubuntu20_runtime
FROM python:3.9.2
USER root

RUN wget https://jsfiles.blob.core.windows.net/ai-models/safety-yolo8.bin
RUN wget https://jsfiles.blob.core.windows.net/ai-models/safety-yolo8.xml
ADD app.py /
ADD runscript.sh /
CMD chmod +x runscript.sh

RUN pip install openvino

CMD [ "python", "/app.py" ]