# %%
#gamma = function of total temperature
tests_imperial_units = {
         '300': {'p0in': 30.0,
                 'T0in': 1039.0},
         '288': {'p0in': 30.0,
                 'T0in': 1518.0} ,
         '293': {'p0in': 30.0,
                 'T0in': 2001.0} ,
         '301': {'p0in': 45.0,
                 'T0in': 1035.0} ,
         '266': {'p0in': 44.8,
                 'T0in': 1503.0} ,
         '294': {'p0in': 49.9,
                 'T0in': 2000.0} ,
         '303': {'p0in': 75.2,
                 'T0in': 1039.0} ,
         '262': {'p0in': 75.2,
                 'T0in': 1518.0} ,
         '290': {'p0in': 75.2,
                 'T0in': 1989.0} ,
         '306': {'p0in': 150.6,
                 'T0in': 1028.0} ,
         '268': {'p0in': 150.6,
                 'T0in': 1484.0} ,
         '311': {'p0in': 253.7,
                 'T0in': 1030.0} ,
         '275': {'p0in': 254.0,
                 'T0in': 1513.0} ,
         '313': {'p0in': 201.7,
                 'T0in': 1517.0} ,
         '315': {'p0in': 74.6,
                 'T0in': 1516.0} ,
         '246': {'p0in': 75.2,
                 'T0in': 1500.0} ,
         '234': {'p0in': 75.2,
                 'T0in': 1527.0} ,
         '276': {'p0in': 202.2,
                 'T0in': 1515.0} ,
         '278': {'p0in': 254.0,
                 'T0in': 518} ,
         '314': {'p0in': 151.7,
                 'T0in': 1506.0} 
        }

# %%
def psia2pascal(psia):
    return psia*6894.76

def rankine2kelvin(rankine):
    return rankine*0.555556

# %%
tests_si_units = {}
for test,val in tests_imperial_units.items():
    tests_si_units[test]={}
    for prop in val:
        if prop == 'p0in':
            tests_si_units[test][prop] = psia2pascal(tests_imperial_units[test][prop])
        if prop == 'T0in':
            tests_si_units[test][prop] = rankine2kelvin(tests_imperial_units[test][prop])

# %%
pressures = [el['p0in'] for el in tests_si_units.values()]
Temperatures = [el['T0in'] for el in tests_si_units.values()]

# %%
min(pressures)/1e6

# %%
max(pressures)/1e6

# %%
min(Temperatures)

# %%
max(Temperatures)

# %%
tests_si_units

# %%
tests_si_units['266']

# %%



