import os, io, boto3, json, csv, urllib
import numpy as np
from PIL import Image, ImageOps

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
sagemaker= boto3.client('runtime.sagemaker')
s3 = boto3.client('s3')


height = int( os.environ['ImageHeight'] )
width = int( os.environ['ImageWidth'] )
color_space = "grayscale"
n_channels = int( os.environ['ImageNumberofChannels'] )

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    inputbucket = event['Records'][0]['s3']['bucket']['name']
    inputkey = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
      s3response = s3.get_object(Bucket=inputbucket, Key=inputkey)
      print("CONTENT TYPE: " + s3response['ContentType'])
      img = Image.open(io.BytesIO(s3response['Body'].read()))
      img = img.resize((height, width), Image.NEAREST)
      img = ImageOps.grayscale(img)
      img = np.array(img)
      img = img[:, :, np.newaxis]
      data_images = np.ndarray((1, height, width, n_channels), dtype=np.uint8)
      data_images[0] = img
      data_images = data_images.astype('float32')
      payload = json.dumps(data_images.tolist())
      smresponse = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                            ContentType='application/json',
                                            Body=payload)
      
      result = json.loads( smresponse['Body'].read().decode() )
      pred = np.array( result["predictions"], dtype=np.float32 )
      uploadresponse = s3.put_object(
        Bucket= os.environ['OuputBucket'],
        Key= '{}.npy'.format( inputkey.split('.')[0] ),
        Body=pred.tobytes()
      )
      print(uploadresponse)
      return uploadresponse
    except Exception as e:
      print(e)
      raise e
