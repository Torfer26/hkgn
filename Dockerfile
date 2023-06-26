# start by pulling the Ubuntu image
FROM ubuntu:latest

# install Python and other dependencies
RUN apt-get update && apt-get install -y python3 python3-pip build-essential libopenblas-dev liblapack-dev gfortran

# set the working directory in the container
WORKDIR /app

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# install the dependencies and packages in the requirements file
RUN pip3 install --no-cache-dir -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

CMD ["view.py"]
