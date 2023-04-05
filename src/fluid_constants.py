import CoolProp.CoolProp as CP

sutherland_constant = {'Air': 110.4}  
sutherland_temperature = {'Air': 273.15}
sutherland_viscosity = {'Air': 1.716e-5}
laminar_prandtl = {'Air': 0.72}
turbulent_prandtl = {'Air': 0.9}

def getFLuidProperties(p0in, T0in, fluid):
    
    gamma = CP.PropsSI('CPMASS', 'P', p0in,'T', T0in, fluid)/ CP.PropsSI('CVMASS', 'P', p0in,'T', T0in, fluid)
    R = CP.PropsSI('GAS_CONSTANT', 'P', p0in,'T', T0in, fluid)/CP.PropsSI('MOLAR_MASS', 'P', p0in,'T', T0in, fluid)
    Tcrit = CP.PropsSI('TCRIT', fluid)
    pcrit = CP.PropsSI('PCRIT', fluid)
    acentricFactor = CP.PropsSI('ACENTRIC', fluid)
    
    return gamma, R, Tcrit, pcrit, acentricFactor