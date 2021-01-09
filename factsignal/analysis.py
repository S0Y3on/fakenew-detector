from konlpy.tag import Kkma
kkma=Kkma()
input_text = '미래통합당 약칭은 통합당이다'

# print(kkma.nouns(input_text))

from konlpy.tag import Hannanum
hannanum = Hannanum()
print(hannanum.nouns(input_text))