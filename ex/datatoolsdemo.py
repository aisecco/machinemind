import tools.datatools as dto

typeno = dto.getTypeOf("192.168.1.1")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("110101198808088888")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("13911111000")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("12.3")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("2021-12-13")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("12:12:12")
print(typeno, dto.getTypeName(typeno))

typeno = dto.getTypeOf("2021-12-13 12:12:12")
print(typeno, dto.getTypeName(typeno))
