#!/usr/bin/env python3

import base64
import hashlib
import json
import os.path
import re
import socket
import sys
import time
import xml.etree.cElementTree as ET
import zlib

from datetime import datetime
from urllib.request import urlopen, HTTPHandler, Request
from urllib.parse import urlencode

argc = len(sys.argv)

if argc != 3:
    sys.exit("Please specify two arguments: <system-name> <job-name>")

cdash_url = 'https://my.cdash.org/submit.php?project=buildtest-cori'
# TODO: make site_name and build_name more configurable.
# For best CDash results, builds names should be consistent (ie not change every time).

site_name = sys.argv[1]
hostname = socket.gethostname()
build_name = sys.argv[2]

input_datetime_format = '%Y/%m/%d %H:%M:%S'
output_datetime_format = '%Y%m%d-%H%M'

build_starttime = None
build_endtime = None

tests = []
report_file = os.path.join(os.getenv("BUILDTEST_ROOT"),"var","report.json")
with open(report_file) as json_file:
    buildtest_data = json.load(json_file)
    for file_name in buildtest_data.keys():
      for test_name, tests_data in buildtest_data[file_name].items():
        test_data = tests_data[0]

        test = {}
        test['name'] = test_name

        state = test_data['state']
        if state == 'PASS':
          test['status'] = 'passed'
        elif state == 'FAIL':
          test['status'] = 'failed'
        else:
          print('Unrecognized state {0} for test {1}; marking it as failed'.format(state, test_name))
          test['status'] = 'failed'

        test['command'] = test_data['command']
        test['path'] = test_data['testpath']
        test['test_content'] = test_data['test_content']
        test['output'] = test_data['output']
        test['output'] += test_data['error']
        test['runtime'] = test_data['runtime']
        test['returncode'] = test_data['returncode']
        test['full_id'] = test_data['full_id']
        test['user'] = test_data['user']
        test['hostname'] = test_data['hostname']
        test["schemafile"] = test_data['schemafile']

        test['tags'] = test_data['tags']
        test['executor'] = test_data['executor']
        test['compiler'] = test_data['compiler']
        test['starttime'] = test_data['starttime']
        test['endtime'] = test_data['endtime']

        # extra stuff to capture as measurements
        full_id = test_data['full_id']
        executor = test_data['executor']

        # tags sound like labels
        tags = test_data['tags']

        # ignored for now
        #testroot = test_data['testroot']
        #stagedir = test_data['stagedir']
        #rundir = test_data['rundir']
        #outfile = test_data['outfile']
        #errfile = test_data['errfile']
        #schemafile = test_data['schemafile']

        # test start and end time.
        starttime = test_data['starttime']
        test_starttime_datetime = datetime.strptime(starttime, input_datetime_format)
        endtime = test_data['endtime']
        test_endtime_datetime = datetime.strptime(endtime, input_datetime_format)
        if not build_starttime or build_starttime > test_starttime_datetime:
          build_starttime = test_starttime_datetime
        if not build_endtime or build_endtime < test_endtime_datetime:
          build_endtime = test_endtime_datetime
        tests.append(test)

build_stamp  = build_starttime.strftime(output_datetime_format)
build_stamp += '-Experimental'

filename = 'Test.xml'
site_element = ET.Element('Site', Name=site_name, BuildName=build_name, BuildStamp=build_stamp)
testing_element = ET.SubElement(site_element, 'Testing')
ET.SubElement(testing_element, 'StartTestTime').text = str(round(datetime.timestamp(build_starttime)))
ET.SubElement(testing_element, 'EndTestTime').text = str(round(datetime.timestamp(build_endtime)))

for test in tests:
  test_element = ET.SubElement(testing_element, 'Test', Status=test['status'])
  ET.SubElement(test_element, 'Name').text = test['name']
  ET.SubElement(test_element, 'Path').text = test['path']
  ET.SubElement(test_element, 'FullCommandLine').text = test['command']
  results_element = ET.SubElement(test_element, 'Results')

  runtime_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='numeric/double', name='Execution Time')
  ET.SubElement(runtime_measurement, 'Value').text = str(test['runtime'])

  testid_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='Test ID')
  ET.SubElement(testid_measurement, 'Value').text = test['full_id']

  returncode_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='numeric/double', name='Return Code')
  ET.SubElement(returncode_measurement, 'Value').text = str(test['returncode'])

  user_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='User')
  ET.SubElement(user_measurement, 'Value').text = test['user']

  hostname_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='Hostname')
  ET.SubElement(hostname_measurement, 'Value').text = test['hostname']

  executor_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='executor')
  ET.SubElement(executor_measurement, 'Value').text = test['executor']

  tags_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='Tags')
  ET.SubElement(tags_measurement, 'Value').text = test['tags']

  testpath_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='testpath')
  ET.SubElement(testpath_measurement, 'Value').text = test['path']

  starttime_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='starttime')
  ET.SubElement(starttime_measurement, 'Value').text = test['starttime']

  endtime_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='endtime')
  ET.SubElement(endtime_measurement, 'Value').text = test['endtime']

  compiler_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='compiler')
  ET.SubElement(compiler_measurement, 'Value').text = test['compiler']

  schema_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/string', name='schemafile')
  ET.SubElement(schema_measurement, 'Value').text = test['schemafile']

  output_measurement = ET.SubElement(results_element, 'Measurement')

  base64_zlib_output = base64.b64encode(zlib.compress(bytes(''.join(test['output']), 'ascii'))).decode('ascii')
  ET.SubElement(output_measurement, 'Value', encoding='base64', compression='gzip').text = base64_zlib_output

  gitlab_job_url = os.getenv('CI_JOB_URL')
  if gitlab_job_url is not None:
    gitlab_link_measurement = ET.SubElement(results_element, 'NamedMeasurement', type='text/link', name='View GitLab CI results')
    ET.SubElement(gitlab_link_measurement, 'Value').text = gitlab_job_url


xml_tree = ET.ElementTree(site_element)
xml_tree.write(filename)

# Compute md5 checksum for the contents of this file.
with open(filename) as xml_file:
  md5sum = hashlib.md5(xml_file.read().encode('utf-8')).hexdigest()

buildid_regexp = re.compile("<buildId>([0-9]+)</buildId>")
with open(filename, 'rb') as f:
  params_dict = {
    'build': build_name,
    'site': site_name,
    'stamp': build_stamp,
    'MD5': md5sum,
  }
  encoded_params = urlencode(params_dict)
  url = "{0}&{1}".format(cdash_url, encoded_params)
  hdrs = {
    'Content-Type': 'text/xml',
    'Content-Length': os.path.getsize(filename)
  }
  request = Request(url, data=f, method='PUT', headers=hdrs)
  with urlopen(request) as response:
    resp_value = response.read()
    if isinstance(resp_value, bytes):
        resp_value = resp_value.decode('utf-8')
    match = buildid_regexp.search(resp_value)
    if match:
      buildid = match.group(1)
      print("Your results have been uploaded to CDash as build #{0}".format(buildid))
