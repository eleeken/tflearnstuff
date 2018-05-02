
price_str = '30.14, 29.58, 26.36, 32.56, 32.82'
type(price_str)

try:
    if not isinstance(price_str, str):
        price_str = str(price_str)
    if isinstance(price_str, int):
        price_str += 1
    elif isinstance(price_str, float):
        price_str += 1.0
    else:
        raise TypeError('price_str is str type!')
except TypeError as err:
    print('Error is:', err)

print('old price_str id = {}'.format(id(price_str)))
price_str = price_str.replace(' ','')
print('old price_str id = {}'.format(id(price_str)))

