temp = {'object': 'error',
        'message': "[{'type': 'missing', 'loc': ('body', 'response_format', 'json_schema', 'name'), 'msg': 'Field required', 'input': {'type': 'object', 'properties': {'completion_message': {'type': 'object', 'properties': {'content': {'type': 'string'}, 'additional_info': {'type': 'string'}}}}}}, {'type': 'extra_forbidden', 'loc': ('body', 'response_format', 'json_schema', 'type'), 'msg': 'Extra inputs are not permitted', 'input': 'object'}, {'type': 'extra_forbidden', 'loc': ('body', 'response_format', 'json_schema', 'properties'), 'msg': 'Extra inputs are not permitted', 'input': {'completion_message': {'type': 'object', 'properties': {'content': {'type': 'string'}, 'additional_info': {'type': 'string'}}}}}]",
        'type': 'BadRequestError', 'param': None, 'code': 400}

message = {'type': 'missing', 'loc': ('body', 'response_format', 'json_schema', 'name'), 'msg': 'Field required',
           'input': {'type': 'object', 'properties': {'completion_message': {'type': 'object', 'properties': {
               'content': {'type': 'string'}, 'additional_info': {'type': 'string'}}}}}}, {'type': 'extra_forbidden',
                                                                                           'loc': (
                                                                                           'body', 'response_format',
                                                                                           'json_schema', 'type'),
                                                                                           'msg': 'Extra inputs are not permitted',
                                                                                           'input': 'object'}, {
    'type': 'extra_forbidden', 'loc': ('body', 'response_format', 'json_schema', 'properties'),
    'msg': 'Extra inputs are not permitted', 'input': {'completion_message': {'type': 'object', 'properties': {
        'content': {'type': 'string'}, 'additional_info': {'type': 'string'}}}}}
