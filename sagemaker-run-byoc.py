# Import Modules that are needed by this code
import boto3
import re
import sagemaker as sage
from time import gmtime, strftime
import os
import numpy as np
import pandas as pd
import itertools
import io

# Define IAM role
role = 'arn:aws:iam::532109487980:role/sagemaker_full_access'

# Set up SageMaker Session and upload the data directory to associated bucket 
sess = sage.Session()
source_bucket_uri = 's3://iris-docker-data/'

# Get account and region to create the image 
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/iris:latest'.format(account, region)

# Set up SageMaker Estimator and fit the training job
model = sage.estimator.Estimator(image,
                      role, 1, 'ml.c4.2xlarge',
                      output_path="s3://{}/output".format(sess.default_bucket()),
                      sagemaker_session=sess)
model.fit(source_bucket_uri)