import glob
import pandas as pd
import tarfile
import urllib.request

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
)

@component(
    packages_to_install=['pandas==1.1.4'],
    output_component_file='component.yaml'
)
def download_and_merge_csv(url: str, output_csv: str):
  with urllib.request.urlopen(url) as res:
    tarfile.open(fileobj=res, mode="r|gz").extractall('data')
  df = pd.concat(
      [pd.read_csv(csv_file, header=None) 
       for csv_file in glob.glob('data/*.csv')])
  df.to_csv(output_csv, index=False, header=False)

#download_and_merge_csv(
#    url='https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz', 
#    output_csv='merged_data.csv')

web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/web/Download/component-sdk-v2.yaml')
