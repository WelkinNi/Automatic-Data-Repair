class Select:
    select_attribute = []
    where_clause = []
    def __init__(self,select_attribute,from_table,where_caluse):
        self.select_attribute = select_attribute
        self.from_table = from_table
        self.where_clause = where_caluse

class Join:
    def __init__(self,table1,table2,join_key,project_attri,select,fd1,fd2):
        self.table1 = table1
        self.table2 = table2
        self.join_key = join_key
        self.project_attri = project_attri
        self.select = select
        self.fd1 = fd1
        self.fd2 = fd2