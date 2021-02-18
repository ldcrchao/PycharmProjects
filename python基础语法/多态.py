# %% 汽车示例
class car():
    '''定义对象'''
    def __init__(self,brand,type,year,mileage,gastank):
        self.brand = brand     #汽车品牌
        self.type = type     #汽车型号
        self.year = year       #汽车生产年份
        self.mileage = mileage #汽车里程数
        self.gastank =gastank  #汽车油量百分比

    '''定义方法'''
    def info(self):  #描述汽车整体信息的方法
        print("my Car's brand is" + ' ' + self.brand + ' ' + self.type + ' ' + 'which was made in ' + str(self.year))
        print(f"\nAnd this car has run {self.mileage} miles , now the oil is {self.gastank} %")


    def gasfilltank(self): #根据油量提示不同的信息
        if self.gastank <=10 :
           print(f"Warning! Your Car's oil only {self.gastank} % ,please add oil !")
        else:
            if self.gastank <=80 :
                print(f"Message! Your Car's oil have {self.gastank} % ,which is not very bad ! ")
            else:
                print(f"Message! Your Car's oil is {self.gastank} % ,which is in a good state !")


    def update(self,mileyears):   #修改汽车里程数的方法
        if mileyears >= self.mileage :
           self.mileage = mileyears  #如果汽车年龄已经被初始化定义，使用update方法进行修改时至少大于初始定义，否则返回下列信息
           print(f"your Car's runmiles is {self.mileage} miles")
        else:
            print(f"your Car's runmiles is {self.mileage} miles at least , cant'be {mileyears} miles")

    def increase(self,miles):   #定义方法增加里程数
        self.mileage = self.mileage + miles

# 继承示例 : 定义1个电动汽车类electric_car，其包含了所有car中的信息
'''electric_car作为子类，car作为父类'''
class electric_car(car): #继承第一步 : 圆括号参数为父类

    # def  __init__(self):
    #     super().__init__() #父类有参数 子类继承必须也得有参数
    #     self.battery = Battery()
    '''定义对象,区别在于子类需要先声明继承父类'''
    def __init__(self,brand,type,year,mileage,gastank,battery):    #继承第2步初始化父类的属性
        super().__init__(brand,type,year,mileage,gastank)  #该命令使得子类继承父类
        #self.battery_size = 70                          #定义电动汽车独有的属性电池容量，原本的电池容量属性属于electric_car类，现转移至Battery类
        if isinstance(battery,Battery) :
           self.battery = battery                         #电动汽车类关联电池类(不是继承)：将电池属性给了电动汽车的一个self参数
        else:
            self.battery = Battery() # 多态 传进来的参数battery可以是任何类型的
    '''定义方法'''
    def describe_battery_size(self): #描述电池容量的方法
        print(f"Now your Car's battery_size is {self.battery.battery_size} Kwh")

    def gasfilltank(self):  #重写父类的油箱函数
        print("The electric Car don't have a gas tank ! ")  #当电动汽车调用此方法时给予提示
    def usefathermethod(self):
        self.increase(200)
        print(self.mileage) # self 调用父类方法
        super().info() # 也可以使用super使用父类方法

# 分类示例 ，随着electric_car的属性越来越多，如电瓶余量、电池寿命、充电次数等属性，但是又有大量的属性可以归类为某个属性
# 如电池寿命属性下的子属性 : 电池充放电次数、电磁电阻磨损、电池电容磨损、电池外壳损耗等等
# 现在定义一个新的子类Battery :
class Battery():

    '''定义对象'''
    def __init__(self,battery_size=70,battery_res=None,battery_c=None): #父类、子类都是可以在初始化实例时可以赋予参数，但是子子类必须先设定好
        self.battery_size = battery_size  #电池容量
        self.battery_res = battery_res    #电池电阻磨损
        self.battery_c = battery_c        #电池电容磨损

    '''定义方法'''
    def describe_battery_size(self): # 述电池容量的方法
        print(f"The Car's battery size is {self.battery_size} Kwh")

    def describe_battery_res(self): # 描述电池电阻磨损的方法
        print(f"The Car's battery res is {self.battery_res} Ω")

    def describe_battery_c(self):  # 描述电池电容磨损的方法
        print(f"The Car's battery c is {self.battery_c} F")

    def get_range(self):    # 描述电池续航里程的方法
        if self.battery_size == 70 :
            range = 240
        elif self.battery_size  == 85 :
            range = 270
        else:
            range = 1000
        message = "This Car can go approximately " + str(range) +' ' + "miles on a full charge"
        print(message)

#%%
if __name__ == '__main__':
    '''汽车car调试代码'''
    mycar = car('audi','A4',2007,20,5)
    print(mycar.info() )
    #直接修改汽车型号属性值
    mycar.type = 'A6'
    print(f'my Car is {mycar.type}'+f' has run {mycar.mileage} miles' )
    #调用方法修改汽车里程属性值
    mycar.update(19) #提示 : your Car's runmiles is 20 miles at least , cant'be 19 miles
    mycar.update(25)
    #调用方法增加汽车里程数
    mycar.increase(10) #查看属性可以发现里程数增加了10
#%%
    '''电动汽车electric_car调试代码'''
    battery = Battery(-160)
    my_electric_car = electric_car('baoma', '3系', 2020, 0 , 90, battery)
    my_electric_car.describe_battery_size()
    my_battery = my_electric_car.battery # 完全继承了battery类 不过在electric_car变成实例化对象的形参
    my_battery.battery_size = 100 # 外部实例化修改属性
    my_electric_car.battery = my_battery # 把这个实例化的类 送入形参 self.battery = Battery()
    print(f'your Car battery_size is {my_electric_car.battery.battery_size} Kwh')  # 直接查看属性得到电池容量
    my_electric_car.usefathermethod()
    #%%
    my_electric_car.describe_battery_size()   # 调用方法查看电池容量
    my_electric_car.battery.describe_battery_size()   #因为electric_car的方法被转移至Battery类，所以也可以在子子类调用方法
    mycar.gasfilltank()                 # 查看父类car的有油量提示
    my_electric_car.gasfilltank()       # 子类电动汽车没有油箱，在子类中已被重写禁用
#%%
    '''电动汽车的子子类属性'''
    my_electric_car.battery.battery_res = 0.1  # Battery的电容电阻参数默认为None，需要先通过直接修改属性值进行赋值
    my_electric_car.battery.battery_c = 0.01
    print(my_electric_car.battery.battery_res ) #直接查看电池电阻磨损属性值
    my_electric_car.battery.describe_battery_res() #通过方法查看电池电阻磨损属性值
    my_electric_car.battery.describe_battery_c() #通过方法查看电池电容磨损属性值
    my_electric_car.battery.get_range()  ##通过方法查看电池续航里程






