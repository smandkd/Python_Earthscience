for byte_arr in n_data:
    for byte in byte_arr:
        if byte != b'':
            ty_name = byte.decode('utf-8'))
            
        if ty_name:
            names.append(ty_name)

위 스니펫은 'n_data'의 모든 바이트를 개별 문자열로 'name'에  추가합니다. 따라서 '['N', 'O', ...]' 와 같은 형태의 리스트가 생성된다. 

for byte_arr in n_data:
    current_name = ''.join(byte.decode('utf-8') for byte in byte_arr if byte !=b'')

    if current_name:
        names.append(current_name)

위 스니펫은 각 'byte_arr'를 하나의 문자열로 결합하여 'names'에 추가한다. 예를 들어, 'n_data'가 [[b'N', b'O'] , [b'T', b'E', b'S']]일 경우, 결과는 '['NO', 'TEST']'와 같이 된다. 