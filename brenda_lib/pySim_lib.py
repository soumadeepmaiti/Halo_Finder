import numpy as np

def read_header_gadget4(data = None, header_fmt = '=QQQQdddddiiqq', fname = None, npt_only = False):
        """ADAPTED FROM BACCO
        Read header when the sym type is gadget4
        
        :param data: the chunk of bytes that contains the header
        :type data: bytes
        
        :param header_fmt: a string representing the format of the header
        :type header_fmt: string

        :param fname: a string with the root of the file name of the snapshots
        :type fname: string

        :param npt_only: a flag for only reading the number of particles
        :type npt_only: bool

        :return: the header in form of dictionary
        :rtype: dict
        """
        N_TYPE = 2 #HARDCODED!!!! ME
        import struct
        import os
        
        #If data is not given, it reads the buffer from fname
        if data is None:
            if fname is None:
                raise RuntimeError("You must give as an input either data or fname")
            if not os.path.exists(fname):
                fname = fname + '.0'
                if not os.path.exists(fname):
                    raise RuntimeError("header file not found {0}".format(fname))
            fd = open(fname, "rb")
            header_fmt = '=QQQQdddddiiqq'
            header_size = struct.calcsize(header_fmt)
            if header_size != struct.unpack('I', fd.read(4))[0]:
                raise ValueError('There is something wrong with the reading of the header: the expected size does not match the read one')
            data = fd.read(header_size)
            fd.close()

        """Create a GadgetHeader from a byte range read from a file."""
        #Initialize variables
        npart = np.zeros(N_TYPE, dtype=np.uint64)
        mass = np.zeros(N_TYPE, dtype=np.double)
        time = 0.
        redshift = 0.
        npartTotal = np.zeros(N_TYPE, dtype=np.uint64)
        num_files = 0
        BoxSize = 0.
        dummy = 1
        Ntrees: np.int64 = 0
        Ntreestotal: np.int64 = 0
        #Read variables from the buffer (data)
        (npart[0],
         npart[1],
         npartTotal[0],
         npartTotal[1],
         mass[0],
         mass[1],
         time,
         redshift,
         BoxSize,
         num_files,
         dummy,
         Ntrees,
         Ntreestotal) = struct.unpack(header_fmt,
                               data)

        if npt_only:
            return npart[1]
        
        #Cosmology is not included in the header and should be read from the cosmology suite
        header = {"BoxSize": BoxSize,
                  "DMFile": fname,
                  "Redshift": redshift,
                  "Time": time,
                  "NpartTotal": npartTotal[1],
                  "NpartHere": npart[1],
                  "ParticleMass": mass[1],
                  "Nfiles": num_files,
                  "WhichSpectrum": 1,
                  "UseRadiation": 0,
                  }
        #UnitLength_in_cm = 3.085678e+24
        #Mpc_norm = header['UnitLength_in_cm'] / 3.085678e+24
        #header['BoxSize'] *= self._Mpc_norm
        return header
        
def read_dark_matter_gadget4(fname, load_vel=True, load_ids=True):
        """Private. Load the dark matter from gadget 4 file format.

        :return: Positions, velocities and IDs of the dark matter particles
        :rtype: tuple
        """
        import sys
        import array
        import struct
        import os
        
        #First of all we read the header with the info that regard all parts of the snapshot
        header_fmt = '=QQQQdddddiiqq'
        header_size = struct.calcsize(header_fmt)
        header = read_header_gadget4(fname=fname, header_fmt=header_fmt)
        
        print(
            "Allocating memory for {0} particles {1}Mb".format(
                header['NpartTotal'],
                header['NpartTotal'] * 3 * 4 / 1024.**2))
        
        #Initialize vectors for pos, vel and ids
        pos = np.zeros((header['NpartTotal'], 3), dtype=np.float32)
        if load_vel:
            vel = np.zeros((header['NpartTotal'], 3), dtype=np.float32)
        else:
            vel = []
        if load_ids:
            ids = np.zeros((header['NpartTotal']), dtype=np.int64)
        else:
            ids = []

        #Let's now iterate on the snapshot files (one per processor used for running the sim)
        istart = 0
        for ifile in range(0, header['Nfiles']):

            if header['Nfiles'] == 1:
                ffname = fname
            else:
                ffname = fname + ('.%d' % ifile)

            if os.path.isfile(ffname) is False:
                raise RuntimeError("file {0} not found".format(ffname))

            with open(ffname, mode='rb') as fin:
                #We read the file block by block
                #1: Header
                hsize1 = struct.unpack('I', fin.read(4))
                dummy_header = fin.read(header_size)
                #We already have the general header, but we need to read the number of part in each file
                npts_here = struct.unpack(header_fmt, dummy_header)[1]
                nptot = struct.unpack(header_fmt, dummy_header)[3]
                hsize2 = struct.unpack('I', fin.read(4))
                #The first and 4 bytes of the block are the size of the block
                #and can be used to check if the block has been read correctly
                assert hsize1 == hsize2

                #2: Particle positions
                hsize1 = struct.unpack('I', fin.read(4))
                npts = int(hsize1[0] / 3 / 4)
                #Let's also check that the number of part in the header 
                #match the ones expected given the size of the buffer
                assert npts == npts_here
                tpos = array.array('f')
                tpos.fromfile(fin, npts * 3)
                tpos = np.reshape(tpos, (npts, 3))
                hsize2 = struct.unpack('I', fin.read(4))
                assert hsize1 == hsize2
                pos[istart:istart + npts_here] = tpos[:npts_here]

                if load_vel:
                    # Velocities
                    hsize1 = struct.unpack('I', fin.read(4))
                    npts = int(hsize1[0] / 3 / 4)
                    assert npts == npts_here
                    tvel = array.array('f')
                    tvel.fromfile(fin, npts * 3)
                    tvel = np.reshape(tvel, (npts, 3))
                    hsize2 = struct.unpack('I', fin.read(4))
                    assert hsize1 == hsize2
                    vel[istart:istart + npts_here] = tvel[:npts_here]

                if load_ids:
                    if not load_vel:
                        hsize1 = struct.unpack('I', fin.read(4))
                        dummy = fin.read(hsize1[0])
                        hsize2 = struct.unpack('I', fin.read(4))
                        assert hsize1 == hsize2
                    # IDs
                    hsize1 = struct.unpack('I', fin.read(4))
                    if npts_here == int(hsize1[0] / 8):
                        ids_fmt = 'Q'
                    elif npts_here == int(hsize1[0] / 4):
                        ids_fmt = 'I'
                    else:
                        raise AssertionError('Something is wrong with the ids. \
                                             Their size appears to be of {} bytes with {} particles'.format(npts_here/hsize1[0], npts_here))
                    tids = array.array(ids_fmt)
                    tids.fromfile(fin, npts)
                    hsize2 = struct.unpack('I', fin.read(4))
                    assert hsize1 == hsize2
                    ids[istart:istart + npts_here] = tids[:npts_here]

                istart += npts_here
                print('Read data in file #{2} for {0}/{1} particles...'.format(npts, int(nptot), ifile))
                
        if load_vel:
            vel = vel * np.sqrt(header['Time'])

        return {'header': header,
                'pos': pos, 
                'vel': vel, 
                'ids': ids}

def get_boxsize(sim_idx=0, snap=0, sim_suite = 'Columbus'):
    if sim_suite == 'Columbus':
        try:
            header = read_header_gadget4(fname='/ptmp/anruiz/Columbus_{0}_A/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_A_00{1}'.format(sim_idx, snap))
            return header['BoxSize']
        except:
            box = np.loadtxt('/u/mesposito/Projects/velocity-power-spectrum/Simulations_info/Columbus_boxsizes_Mpc_h.txt')
            return box[sim_idx]
    else:
        header = read_header_gadget4(fname='/ptmp/anruiz/CassandraMass_1500/CassandraMass_L2800_{0}/snapshots/snapdir_00{1}/snapshot_CassandraMass_L2800_{0}_00{1}'.format('A', snap))
        return header['BoxSize']

def get_expfactor(sim_idx, snap, sim_suite = 'Columbus', L=2000):
    if sim_suite == 'Columbus':
        try:
            header = read_header_gadget4(fname='/ptmp/anruiz/Columbus_{0}_A/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_A_00{1}'.format(sim_idx, snap))
        except:
            header = read_header_gadget4(fname='/ptmp/mesposito/Columbus_{0}_A/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_A_00{1}'.format(sim_idx, snap))
    elif sim_suite == 'Small sims':
        header = read_header_gadget4(fname='/ptmp/mesposito/CosmoSims/Col_{}_N512_L{}.0_output/0.00/snapdir_{:03}/snapshot_{:03}'.format(sim_idx, L, snap, snap))
    return 1/(1+header['Redshift'])
    
def read_Columbus_sim(sim_idx, snap, letter, load_vel=True, load_ids=True):
    try:
        fname = '/ptmp/anruiz/Columbus_{0}_{2}/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_{2}_00{1}'.format(sim_idx, snap, letter)
        return read_dark_matter_gadget4(fname, load_vel=load_vel, load_ids=load_ids)
    except:
        fname = '/ptmp/mesposito/Columbus_{0}_{2}/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_{2}_00{1}'.format(sim_idx, snap, letter)
        return read_dark_matter_gadget4(fname, load_vel=load_vel, load_ids=load_ids)