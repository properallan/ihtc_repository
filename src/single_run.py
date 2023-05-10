from tracemalloc import Snapshot
from train_nn_parts import *
from solvers import *
import meshio
import time

def Area(x, xt):
    # Area function as defined in Yeom 
    ret = np.zeros_like(x)
    for i,xi in enumerate(x):
        if xi < xt:
            ret[i] = 2.5 + 3.0*(x[i]/xt - 1.5)*(x[i]/xt)**2.0
        elif xi >= xt:
            ret[i] = 3.5 - (x[i]/xt)*(6.0 - 4.5*(x[i]/xt) + (x[i]/xt)**2.0)
    return ret

class ROM(object):
    def __init__(self, q1d_dataset, su2_dataset, rank=None):

        self.q1dSVD = SVD(q1d_dataset)
        self.su2SVD = SVD(su2_dataset)
        
        self.bounds = [0,1]
        
        self.q1dSVD.normalize(bounds=self.bounds)
        self.su2SVD.normalize(bounds=self.bounds)
        
        self.q1dSVD.subtractMean()
        self.su2SVD.subtractMean()

        self.q1dSVD.SVD()
        self.su2SVD.SVD()

        self.setRank(rank)

    def setRank(self, rank):
        self.rank = rank
        self.q1dSVD.setRank(self.rank)
        self.su2SVD.setRank(self.rank)

    def setEnergy(self, e):
        self.energyPreserved = e
        self.rank = self.q1dSVD.findRank(self.energyPreserved)
        self.su2SVD.setRank(self.rank)
        
    def loadNN(self, file):
        self.nn = tf.keras.models.load_model(file) 
    
    def reduce(self, snapshot):
        snapshot,min,max = self.q1dSVD.normalize(snapshot, [0,1], self.q1dSVD.min, self.q1dSVD.max)
        snapshot,mean = self.q1dSVD.subtractMean(snapshot, self.q1dSVD.mean)
        L = (self.q1dSVD.u.T @ snapshot).T

        return L

    def reconstruct(self, L):
        nnL = self.nn.predict(L)
        snapRec = self.su2SVD.reconstruct(nnL)

        return snapRec

        
class FOM(object):
    # Setup nozzle shape 
    nozzleL = [0.0,10.0]
    xt = 5.0

    # Boundary conditions
    T0in = 291.3
    pr = 0.89
    p0in = 104074.0
    Min = 0.01
    
    # Fluid properties
    gamma = 1.4
    R = 287.0
    
    # eulerQ1D setup
    nQ1D = 301
    itmaxQ1D = 500000
    itprintQ1D = 1000
    CFLQ1D = 0.1
    tolQ1D = 1e-8
    tschemeQ1D = 'RK4'
    fschemeQ1D = 'AUSM'
    dttypeQ1D = 'Global'
    dimQ1D = 'Dimensionless'

    # SU2 setup
    nxSU2 = 201
    nySU2 = 46
    itmaxSU2 = 10000
    itprintSU2 = 10
    CFLSU2 = 100
    tolSU2 = 1e-8
    tschemeSU2 = 'EULER_IMPLICIT'
    fschemeSU2 = 'ROE'
    dimSU2 = 'DIMENSIONAL'

    noz = nozzle()

    def __init__(self, path): 
        self.path = path
        
    def setup(self):
        self.noz.setBC(self.p0in, self.T0in, self.Min, self.pr*self.p0in)
        self.noz.setFluid(self.R, self.gamma)

        # Q1D setup
        self.noz.setX(self.nozzleL[0], self.nozzleL[1], self.nQ1D)
        self.noz.setS(Area, self.xt)
        
        self.noz.setQ1DSolver(self.itmaxQ1D, self.itprintQ1D, self.CFLQ1D, self.tolQ1D, self.tschemeQ1D, self.fschemeQ1D, self.dttypeQ1D, self.dimQ1D)
        self.Q1Dpath = self.path + 'Q1D/'
        self.noz.setupQ1D( self.Q1Dpath, 'setupQ1D.txt')
        
        # SU2 setup
        self.noz.setXSU2(self.nozzleL[0], self.nozzleL[1], self.nxSU2)
        self.noz.setSSU2(Area, self.xt)
        self.noz.setNySU2(self.nySU2)

        self.noz.setupSU2Solver(self.itmaxSU2, self.itprintSU2, self.CFLSU2, np.log10(self.tolSU2), self.tschemeSU2, self.fschemeSU2, self.dimSU2)
        self.SU2path = self.path + 'SU2/'
        self.noz.setupSU2( self.SU2path, 'setupSU2.cfg')
        su2Mesh = self.SU2path + 'inputs/setupSU2.su2'
        self.PVMesh = self.SU2path + 'inputs/setupSU2.vtk'
        m = meshio.read(su2Mesh)
        meshio.write(self.PVMesh, m)

    def solveQ1Q(self, solver='./eulerQ1D'):
        self.noz.solveQ1D(solver)

    def solveSU2(self, solver='./SU2_CFD'):
        self.noz.solveSU2(solver)
        self.savePVMesh()


    def getQ1DMach(self):
        Mach = np.loadtxt(self.Q1Dpath + 'outputs/M.txt')
        return np.expand_dims(Mach,axis=1)

    def getSU2Mach(self):
        mesh = pv.read(self.SU2path + 'outputs/solution.vtk')
        mesh.set_active_scalars('Mach')
        Mach = mesh['Mach']
        return np.expand_dims(Mach,axis=1)

    def savePVMesh(self):
        vtkfile = self.SU2path + 'outputs/solution.vtk'
        vtk = pv.read(vtkfile)
        vtk.clear_data()
        dot_idx = vtkfile.rfind('.')
        vtkfile_mesh = vtkfile[:dot_idx]+'_mesh.vtk'
        vtk.save(vtkfile_mesh)
        self.PVMesh = vtkfile_mesh
        return vtkfile_mesh

    def plotSU2(self, title, arr, titleplot):
        pv.set_plot_theme("document")
        p = pv.Plotter()
        p.enable()
        p.enable_anti_aliasing()

        mesh = pv.read(self.PVMesh)
        mesh[title] = arr
        mesh2 = mesh.reflect((0, -1, 0), point=(0, 0, 0))
        mesh2['Mach'] = arr
        mesh = mesh + mesh2
        mesh.set_active_scalars('Mach')
        
        p.add_mesh(mesh, opacity=0.85, render=True, cmap='plasma')
    
        p.set_viewup([0, 1, 0])
        p.fly_to([5,0,0])

        p.set_position([5.0, -0.01, 7.5])
        p.window_size = [1280,480]
        #p.save_graphic('./annSU2_'+str(pod.modes.shape[1])+'_modes.pdf')
        #p.show(title=titleplot, window_size=[1280,480], interactive=False, full_screen=False, interactive_update=False, auto_close=False)
        p.show(title=titleplot, window_size=[1280,480], auto_close=True, interactive=False, interactive_update=False)

def main(xt, pr):
    # setup reduced order model
    rom = ROM('../dataVERYLARGE')
    rom.setEnergy(99.00)
    rom.loadNN('trained_nn1')

    # setup full order models
    fom = FOM('../single_run/')
    #fom.xt = 5.2
    #fom.pr = 0.621
    fom.xt = xt
    fom.pr = pr
    fom.setup()

    start = time.time()
    fom.solveQ1Q()
    Q1DMach = fom.getQ1DMach()
    L = rom.reduce(Q1DMach)
    romMach = rom.reconstruct(L)
    end = time.time()
    fom.plotSU2('Mach', romMach, 'Reconstructed Flow Field from Q1D solution - Time: {:.2} s'.format(end-start))

    start = time.time()
    fom.solveSU2()
    SU2Mach = fom.getSU2Mach()
    end = time.time()

    fom.plotSU2('Mach', SU2Mach, 'Full Order Solution - Time: {:.2} s'.format(end-start))

    Q1DSVDMach = rom.q1dSVD.reconstruct(L)


    xLSU2 = np.linspace(0,1,len(SU2Mach.reshape(46,201)[0,4:197]))
    xLQ1D = np.linspace(0,1,len(Q1DMach))
    plt.plot(xLSU2, SU2Mach.reshape(46,201)[0,4:197], label='SU2')
    plt.plot(xLSU2, romMach.reshape(46,201)[0,4:197], label='NN')
    plt.plot(xLQ1D, Q1DMach, label='Q1D')
    plt.plot(xLQ1D, Q1DSVDMach, ls='--', label='Q1DSVD' )
    plt.legend()
    plt.ylabel('Mach')
    plt.xlabel('x/L')
    plt.show()

if __name__ == '__main__':
    pass
    
    main(5.5, 0.89)