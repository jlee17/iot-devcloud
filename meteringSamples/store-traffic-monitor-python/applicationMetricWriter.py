import xml.etree.ElementTree as ET
import socket
import pickle
import struct
import time
import sys
from threading import Thread
import json

INF_UID = 0
INF_TIME = 0
start = time.time()
data = {}   
data['inferenceStats'] = []
def send_application_metrics(modelXML, targetHardware):
	global data    
	tree = ET.parse(modelXML)
	root = tree.getroot()
	dictLayerTypes = {}
	metrics = ([])
	width = "0"
	height = "0"
	for elem in root:
	    for subelem in elem.findall('layer'):
	        if (subelem.get('type') == "Input"):
	            width = subelem[0][0][2].text
	            height=subelem[0][0][3].text            
	        if (subelem.get('type') not in dictLayerTypes):
	            dictLayerTypes[subelem.get('type')] = 1
	        else:
	            dictLayerTypes[subelem.get('type')] += 1
	            precision = subelem.get("precision")
	data['applicationMetrics'] = []
	data['applicationMetrics'].append({'modelName': root.attrib.get("name")})
	data['applicationMetrics'].append({'precision': str(precision)})
	data['applicationMetrics'].append({'targetHardware': str(targetHardware)})
	print(targetHardware)
	data['applicationMetrics'].append({'numLayers': str(sum(dictLayerTypes.values()) - 1)})
	data['applicationMetrics'].append({'width': str(width)})
	data['applicationMetrics'].append({'height': str(height)})
	data['layerInfo'] = []    
	for layerName in dictLayerTypes:
	    data['layerInfo'].append({layerName : dictLayerTypes[layerName]})
#	data['infrerenceMetrics'] = []
	global INF_TIME
	global INF_UID
	avgInferenceTime=INF_TIME/INF_UID
	data['applicationMetrics'].append({'avgInferenceTime': avgInferenceTime})
	data['applicationMetrics'].append({'inferenceCount': INF_UID})
	data['applicationMetrics'].append({'totalinferenceTime': INF_TIME})
#	time.sleep(30)
	with open('/tmp/AppMetrics.json', 'w') as outfile:  
            json.dump(data, outfile)          

def send_inference_time(infTime):
	global INF_UID, INF_TIME, start
	global data
	elapsed = time.time() - start
	data['inferenceStats'].append({time.time(): 1000/infTime})
	if elapsed > 5:
	    data['inferenceStats'].append({time.time(): 1000/infTime})
	    start = time.time()        
	INF_TIME+=infTime
	INF_UID+=1
