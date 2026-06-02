$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '06/01/2026 19:56:24')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:45')
			I(1, 'ComEngine Memory', '105 M')
		$end 'TotalInfo'
		GroupOptions=8
		TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Executing From\', \'C:\\\\Program Files\\\\ANSYS Inc\\\\v251\\\\AnsysEM\\\\HFSSCOMENGINE.exe\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='HPC'
			$begin 'StartInfo'
				I(1, 'Type', 'Manual')
				I(1, 'Distribution Types', 'Variations, Frequencies, Domain Solver, Transient Excitations, Mesh Assembly')
				I(1, 'MPI Vendor', 'Intel')
				I(1, 'MPI Version', '2021')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions(Memory=8)
			ProfileItem('Two Level', 0, 0, 0, 0, 0, 'I(1, 0, \'Disabled\')', false, true)
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'55.8 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 98.6 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 19:56:24')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:01')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 0, 0, 0, 0, 33272, 'I(2, 1, \'Type\', \'Phi\', 2, \'Tetrahedra\', 626, false)', true, true)
			ProfileItem('Post', 0, 0, 0, 0, 35980, 'I(2, 2, \'Tetrahedra\', 626, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Lambda Refine', 0, 0, 0, 0, 25140, 'I(2, 2, \'Tetrahedra\', 1757, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Simulation Setup', 0, 0, 0, 0, 175116, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 0, 0, 0, 0, 185948, 'I(2, 2, \'Tetrahedra\', 1615, false, 1, \'Disk\', \'33.3 KB\')', true, true)
			ProfileItem('Port Refine', 0, 0, 0, 0, 22620, 'I(2, 2, \'Tetrahedra\', 1839, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 19:56:25')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:20')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 1'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 176796, 'I(2, 2, \'Tetrahedra\', 1685, false, 1, \'Disk\', \'4.01 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 200672, 'I(3, 2, \'Tetrahedra\', 1685, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'3.21 KB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 261424, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 11857, false, 3, \'Matrix bandwidth\', 20.4798, \'%5.1f\', 1, \'Disk\', \'49.5 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 261424, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'723 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103680, 'I(1, 0, \'Adaptive Pass 1\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780332985')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'11857\', 1, \'Matrixbw\', \'20.513399\', 1, \'Matrixnnz\', \'243227\', 1, \'Rootdim\', \'533\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'3007639969595392.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1056\', 1, \'Factornnz\', \'1823729\', 1, \'Factorestflops\', \'994215211\', 1, \'Fbsestflops\', \'6118976\', 1, \'Rootfactestflops\', \'50476020\', 1, \'Rootfbsestflops\', \'142044\', 1, \'Analysistimesec\', \'0.045193\', 1, \'Analysismemkb\', \'7968.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.057779\', 1, \'Factorizationmentotalkb\', \'61499.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.005149\', 1, \'Fbscputimesec\', \'0.005149\', 1, \'Fbsmemorytotalkb\', \'66056.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'66056\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 2'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 24096, 'I(2, 2, \'Tetrahedra\', 2345, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 177968, 'I(2, 2, \'Tetrahedra\', 2135, false, 1, \'Disk\', \'3.62 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 205892, 'I(3, 2, \'Tetrahedra\', 2135, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 274688, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 14849, false, 3, \'Matrix bandwidth\', 20.7091, \'%5.1f\', 1, \'Disk\', \'59.6 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 274688, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'708 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103988, 'I(1, 0, \'Adaptive Pass 2\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780332988')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'14849\', 1, \'Matrixbw\', \'20.734301\', 1, \'Matrixnnz\', \'307883\', 1, \'Rootdim\', \'527\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1815830031499264.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1320\', 1, \'Factornnz\', \'2402855\', 1, \'Factorestflops\', \'1384058122\', 1, \'Fbsestflops\', \'8051765\', 1, \'Rootfactestflops\', \'48790741\', 1, \'Rootfbsestflops\', \'138864\', 1, \'Analysistimesec\', \'0.062529\', 1, \'Analysismemkb\', \'8928.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.066471\', 1, \'Factorizationmentotalkb\', \'79586.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.005581\', 1, \'Fbscputimesec\', \'0.005581\', 1, \'Fbsmemorytotalkb\', \'76100.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'79586\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.214776, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 3'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 24732, 'I(2, 2, \'Tetrahedra\', 2992, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 179508, 'I(2, 2, \'Tetrahedra\', 2731, false, 1, \'Disk\', \'4.01 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 213024, 'I(3, 2, \'Tetrahedra\', 2731, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 300044, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 18755, false, 3, \'Matrix bandwidth\', 20.9339, \'%5.1f\', 1, \'Disk\', \'74.9 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 300044, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'878 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104220, 'I(1, 0, \'Adaptive Pass 3\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780332991')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'18755\', 1, \'Matrixbw\', \'20.954201\', 1, \'Matrixnnz\', \'392996\', 1, \'Rootdim\', \'553\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1424659979436032.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1658\', 1, \'Factornnz\', \'3226802\', 1, \'Factorestflops\', \'2022017895\', 1, \'Fbsestflops\', \'10827311\', 1, \'Rootfactestflops\', \'56373761\', 1, \'Rootfbsestflops\', \'152904\', 1, \'Analysistimesec\', \'0.081592\', 1, \'Analysismemkb\', \'11356.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.095919\', 1, \'Factorizationmentotalkb\', \'107063.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.008165\', 1, \'Fbscputimesec\', \'0.008165\', 1, \'Fbsmemorytotalkb\', \'96196.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'107063\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.23741, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 4'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 25900, 'I(2, 2, \'Tetrahedra\', 3812, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 181132, 'I(2, 2, \'Tetrahedra\', 3489, false, 1, \'Disk\', \'4.01 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 221504, 'I(3, 2, \'Tetrahedra\', 3489, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 339036, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 23707, false, 3, \'Matrix bandwidth\', 21.1234, \'%5.1f\', 1, \'Disk\', \'94.2 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 339036, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'1.07 MB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104316, 'I(1, 0, \'Adaptive Pass 4\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780332994')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'23707\', 1, \'Matrixbw\', \'21.141100\', 1, \'Matrixnnz\', \'501192\', 1, \'Rootdim\', \'655\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2664730049118208.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'2107\', 1, \'Factornnz\', \'4501408\', 1, \'Factorestflops\', \'3222710000\', 1, \'Fbsestflops\', \'15261489\', 1, \'Rootfactestflops\', \'93673864\', 1, \'Rootfbsestflops\', \'214512\', 1, \'Analysistimesec\', \'0.103596\', 1, \'Analysismemkb\', \'14576.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.113328\', 1, \'Factorizationmentotalkb\', \'143157.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.014682\', 1, \'Fbscputimesec\', \'0.014682\', 1, \'Fbsmemorytotalkb\', \'129596.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'143157\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0953279, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 5'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 26920, 'I(2, 2, \'Tetrahedra\', 4864, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 183836, 'I(2, 2, \'Tetrahedra\', 4460, false, 1, \'Disk\', \'4.39 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 233512, 'I(3, 2, \'Tetrahedra\', 4460, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 379492, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 30035, false, 3, \'Matrix bandwidth\', 21.2836, \'%5.1f\', 1, \'Disk\', \'119 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 379492, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'286 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104320, 'I(1, 0, \'Adaptive Pass 5\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780332997')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'30035\', 1, \'Matrixbw\', \'21.297800\', 1, \'Matrixnnz\', \'639678\', 1, \'Rootdim\', \'743\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2224709978554368.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'2659\', 1, \'Factornnz\', \'6099734\', 1, \'Factorestflops\', \'4866970000\', 1, \'Fbsestflops\', \'20798461\', 1, \'Rootfactestflops\', \'136728065\', 1, \'Rootfbsestflops\', \'276024\', 1, \'Analysistimesec\', \'0.133806\', 1, \'Analysismemkb\', \'18524.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.163399\', 1, \'Factorizationmentotalkb\', \'189310.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.012890\', 1, \'Fbscputimesec\', \'0.012890\', 1, \'Fbsmemorytotalkb\', \'161720.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'189310\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0671034, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 6'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 28040, 'I(2, 2, \'Tetrahedra\', 6069, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 186424, 'I(2, 2, \'Tetrahedra\', 5573, false, 1, \'Disk\', \'4.39 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 246116, 'I(3, 2, \'Tetrahedra\', 5573, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'345 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 2, 0, 436320, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 37291, false, 3, \'Matrix bandwidth\', 21.3916, \'%5.1f\', 1, \'Disk\', \'147 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 436320, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'321 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104328, 'I(1, 0, \'Adaptive Pass 6\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333000')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'37291\', 1, \'Matrixbw\', \'21.405899\', 1, \'Matrixnnz\', \'798248\', 1, \'Rootdim\', \'855\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2206040024154112.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'3274\', 1, \'Factornnz\', \'8026180\', 1, \'Factorestflops\', \'6983940000\', 1, \'Fbsestflops\', \'27468705\', 1, \'Rootfactestflops\', \'208346516\', 1, \'Rootfbsestflops\', \'365512\', 1, \'Analysistimesec\', \'0.174765\', 1, \'Analysismemkb\', \'23148.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.195140\', 1, \'Factorizationmentotalkb\', \'245302.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.018618\', 1, \'Fbscputimesec\', \'0.018618\', 1, \'Fbsmemorytotalkb\', \'210036.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'245302\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.034667, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 7'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 29936, 'I(2, 2, \'Tetrahedra\', 7744, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 189332, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'4.01 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265428, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 2, 0, 500312, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 1, \'Disk\', \'187 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 500312, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'394 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104340, 'I(1, 0, \'Adaptive Pass 7\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333003')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2940709883609088.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4133\', 1, \'Factornnz\', \'10959704\', 1, \'Factorestflops\', \'10925600000\', 1, \'Fbsestflops\', \'37649127\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.201040\', 1, \'Analysismemkb\', \'29524.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.267134\', 1, \'Factorizationmentotalkb\', \'327165.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.021085\', 1, \'Fbscputimesec\', \'0.021085\', 1, \'Fbsmemorytotalkb\', \'260984.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'327165\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0188307, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 19:56:45')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:24')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'HPC\', \'Enabled\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Solution - Sweep'
				$begin 'StartInfo'
					I(0, 'Interpolating HFSS Frequency Sweep')
					I(1, 'Time', '06/01/2026 19:56:45')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(1, 'Elapsed Time', '00:00:24')
				$end 'TotalInfo'
				GroupOptions=4
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'From 1GHz to 3GHz, 101 Frequencies\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 3GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #1; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 187988, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265684, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 3, 0, 370428, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'187 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 370428, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333006')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1926920031698944.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.206220\', 1, \'Analysismemkb\', \'29040.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.274879\', 1, \'Factorizationmentotalkb\', \'148995.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000897\', 1, \'Fbscputimesec\', \'0.000897\', 1, \'Fbsmemorytotalkb\', \'130444.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'148995\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #2; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 187980, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265320, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 378576, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 378576, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333008')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2235009981218816.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.051723\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.253487\', 1, \'Factorizationmentotalkb\', \'156767.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000782\', 1, \'Fbscputimesec\', \'0.000782\', 1, \'Fbsmemorytotalkb\', \'140052.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'156767\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 1, Frequency: 3GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 2, Frequency: 1GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #3\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188164, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265368, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 373412, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 373412, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333010')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2553270010642432.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.052831\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.269426\', 1, \'Factorizationmentotalkb\', \'156229.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000786\', 1, \'Fbscputimesec\', \'0.000786\', 1, \'Fbsmemorytotalkb\', \'133748.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'156229\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 141.822%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106420, 'I(1, 0, \'Frequency #3;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.5GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #4\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188020, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 264776, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 378664, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 378664, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333012')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2483640000839680.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.053604\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.256183\', 1, \'Factorizationmentotalkb\', \'161265.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000855\', 1, \'Fbscputimesec\', \'0.000855\', 1, \'Fbsmemorytotalkb\', \'139564.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'161265\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error =  85.303%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106684, 'I(1, 0, \'Frequency #4;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.25GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #5\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188144, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265056, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 375264, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 375264, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333015')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2641729962377216.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.054188\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.269491\', 1, \'Factorizationmentotalkb\', \'163201.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000934\', 1, \'Fbscputimesec\', \'0.000934\', 1, \'Fbsmemorytotalkb\', \'135772.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'163201\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error =  64.274%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106848, 'I(1, 0, \'Frequency #5;  Interpolating frequency sweep\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 2.4GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 2.4GHz; S Matrix Error =  23.972%\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.7GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #6\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 187352, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265368, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 373216, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 373216, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333017')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2992699925856256.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.054516\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.287329\', 1, \'Factorizationmentotalkb\', \'162437.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001220\', 1, \'Fbscputimesec\', \'0.001220\', 1, \'Fbsmemorytotalkb\', \'133472.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'162437\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 2.7GHz; S Matrix Error =   2.719%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106944, 'I(1, 0, \'Frequency #6;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.55GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #7\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188108, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 264948, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 372292, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 372292, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333019')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1792929936965632.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.053493\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.258887\', 1, \'Factorizationmentotalkb\', \'159921.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000682\', 1, \'Fbscputimesec\', \'0.000682\', 1, \'Fbsmemorytotalkb\', \'132992.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'159921\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.55GHz; S Matrix Error =   0.397%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107024, 'I(1, 0, \'Frequency #7;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.85GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #8\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 187804, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 264556, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 362860, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 362860, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333021')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2297320024571904.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.055085\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.272296\', 1, \'Factorizationmentotalkb\', \'151039.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000744\', 1, \'Fbscputimesec\', \'0.000744\', 1, \'Fbsmemorytotalkb\', \'124252.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'151039\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.85GHz; S Matrix Error =   0.155%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107032, 'I(1, 0, \'Frequency #8;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.2GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #9\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188020, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265724, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 372552, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 372552, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333023')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1885160064679936.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.052317\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.274272\', 1, \'Factorizationmentotalkb\', \'152219.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000705\', 1, \'Fbscputimesec\', \'0.000705\', 1, \'Fbsmemorytotalkb\', \'132476.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'152219\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 2.2GHz; S Matrix Error =   0.128%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107036, 'I(1, 0, \'Frequency #9;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.1GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #10\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188216, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 265940, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 366680, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 366680, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333025')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2032280008654848.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.053557\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.263295\', 1, \'Factorizationmentotalkb\', \'160506.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.000692\', 1, \'Fbscputimesec\', \'0.000692\', 1, \'Fbsmemorytotalkb\', \'126352.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'160506\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 2.1GHz; S Matrix Error =   0.051%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107040, 'I(1, 0, \'Frequency #10;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.75GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #11\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 187256, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 264412, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 2, 0, 367828, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 367828, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333028')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2372309885124608.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.054533\', 1, \'Analysismemkb\', \'26299.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.261952\', 1, \'Factorizationmentotalkb\', \'156594.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001105\', 1, \'Fbscputimesec\', \'0.001105\', 1, \'Fbsmemorytotalkb\', \'128908.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'156594\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 12, Frequency: 1.75GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107132, 'I(1, 0, \'Frequency #11;  Interpolating frequency sweep\')', true, true)
				ProfileFootnote('I(1, 0, \'Interpolating sweep converged and is passive\')', 0)
				ProfileFootnote('I(1, 0, \'HFSS: Interpolating sweep\')', 0)
			$end 'ProfileGroup'
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Simulation Summary'
			$begin 'StartInfo'
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'98.6 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:01\', 1, \'Total Memory\', \'217 MB\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:20\', 1, \'Average memory/process\', \'489 MB\', 1, \'Max memory/process\', \'489 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:24\', 1, \'Average memory/process\', \'363 MB\', 1, \'Max memory/process\', \'370 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 7116, false, 2, \'Max matrix size\', 47359, false, 1, \'Matrix bandwidth\', \'21.5\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'06/01/2026 19:57:09\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '06/01/2026 20:00:07')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:33')
			I(1, 'ComEngine Memory', '104 M')
		$end 'TotalInfo'
		GroupOptions=8
		TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Executing From\', \'C:\\\\Program Files\\\\ANSYS Inc\\\\v251\\\\AnsysEM\\\\HFSSCOMENGINE.exe\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='HPC'
			$begin 'StartInfo'
				I(1, 'Type', 'Manual')
				I(1, 'Distribution Types', 'Variations, Frequencies, Domain Solver, Transient Excitations, Mesh Assembly')
				I(1, 'MPI Vendor', 'Intel')
				I(1, 'MPI Version', '2021')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions(Memory=8)
			ProfileItem('Two Level', 0, 0, 0, 0, 0, 'I(1, 0, \'Disabled\')', false, true)
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'55.7 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 97.8 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 20:00:07')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:09')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 7'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.46GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 189620, 'I(2, 2, \'Tetrahedra\', 7116, false, 1, \'Disk\', \'4.01 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 264560, 'I(3, 2, \'Tetrahedra\', 7116, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'35.3 KB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 3, 0, 507252, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 47359, false, 3, \'Matrix bandwidth\', 21.4909, \'%5.1f\', 1, \'Disk\', \'187 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 507252, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'228 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 102500, 'I(1, 0, \'Adaptive Pass 7\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333208')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'47359\', 1, \'Matrixbw\', \'21.503000\', 1, \'Matrixnnz\', \'1018362\', 1, \'Rootdim\', \'993\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1465100049317888.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'4128\', 1, \'Factornnz\', \'10949630\', 1, \'Factorestflops\', \'10904800000\', 1, \'Fbsestflops\', \'37654456\', 1, \'Rootfactestflops\', \'326387595\', 1, \'Rootfbsestflops\', \'493024\', 1, \'Analysistimesec\', \'0.206147\', 1, \'Analysismemkb\', \'29344.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.279256\', 1, \'Factorizationmentotalkb\', \'323350.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.019358\', 1, \'Fbscputimesec\', \'0.019358\', 1, \'Fbsmemorytotalkb\', \'268224.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'323350\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 8'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.46GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 32704, 'I(2, 2, \'Tetrahedra\', 9883, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 194536, 'I(2, 2, \'Tetrahedra\', 9097, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 289452, 'I(3, 2, \'Tetrahedra\', 9097, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 4, 0, 591724, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 60273, false, 3, \'Matrix bandwidth\', 21.572, \'%5.1f\', 1, \'Disk\', \'237 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 591724, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'476 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103204, 'I(1, 0, \'Adaptive Pass 8\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333211')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'60273\', 1, \'Matrixbw\', \'21.582800\', 1, \'Matrixnnz\', \'1300859\', 1, \'Rootdim\', \'1089\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2052580037361664.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'5287\', 1, \'Factornnz\', \'14689435\', 1, \'Factorestflops\', \'16211500000\', 1, \'Fbsestflops\', \'50429858\', 1, \'Rootfactestflops\', \'430494936\', 1, \'Rootfbsestflops\', \'592960\', 1, \'Analysistimesec\', \'0.274936\', 1, \'Analysismemkb\', \'37132.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.363142\', 1, \'Factorizationmentotalkb\', \'430962.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.026653\', 1, \'Fbscputimesec\', \'0.026653\', 1, \'Fbsmemorytotalkb\', \'335360.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'430962\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0259432, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 9'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.46GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 34192, 'I(2, 2, \'Tetrahedra\', 12314, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 199696, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315368, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'345 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 6, 0, 720368, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 1, \'Disk\', \'294 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 720368, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'547 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103208, 'I(1, 0, \'Adaptive Pass 9\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333215')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1509800055668736.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6507\', 1, \'Factornnz\', \'19751204\', 1, \'Factorestflops\', \'24488600000\', 1, \'Fbsestflops\', \'68041534\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.341197\', 1, \'Analysismemkb\', \'47060.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.551652\', 1, \'Factorizationmentotalkb\', \'567238.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.036990\', 1, \'Fbscputimesec\', \'0.036990\', 1, \'Fbsmemorytotalkb\', \'446520.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'567238\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0152028, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 20:00:16')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:23')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'HPC\', \'Enabled\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Solution - Sweep'
				$begin 'StartInfo'
					I(0, 'Interpolating HFSS Frequency Sweep')
					I(1, 'Time', '06/01/2026 20:00:16')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(1, 'Elapsed Time', '00:00:23')
				$end 'TotalInfo'
				GroupOptions=4
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'From 1GHz to 3GHz, 101 Frequencies\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 3GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #1; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196756, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315024, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 508312, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'294 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 508312, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333218')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2952479973048320.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.347809\', 1, \'Analysismemkb\', \'47072.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.481845\', 1, \'Factorizationmentotalkb\', \'263680.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001046\', 1, \'Fbscputimesec\', \'0.001046\', 1, \'Fbsmemorytotalkb\', \'234672.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'263680\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #2; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196792, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315264, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 474144, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 474144, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333220')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2914569907339264.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.087293\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.495531\', 1, \'Factorizationmentotalkb\', \'252089.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001091\', 1, \'Fbscputimesec\', \'0.001091\', 1, \'Fbsmemorytotalkb\', \'200548.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'252089\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 1, Frequency: 3GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 2, Frequency: 1GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #3\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196316, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 314668, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 4, 0, 477580, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 477580, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333222')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2572670109483008.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.086764\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.472610\', 1, \'Factorizationmentotalkb\', \'247571.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001185\', 1, \'Fbscputimesec\', \'0.001185\', 1, \'Fbsmemorytotalkb\', \'204356.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'247571\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 142.225%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104984, 'I(1, 0, \'Frequency #3;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.5GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #4\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196536, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315396, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 4, 0, 489084, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 489084, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333224')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2314879998361600.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.087215\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.473830\', 1, \'Factorizationmentotalkb\', \'257052.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001446\', 1, \'Fbscputimesec\', \'0.001446\', 1, \'Fbsmemorytotalkb\', \'215092.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'257052\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error =  86.154%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105128, 'I(1, 0, \'Frequency #4;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.25GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #5\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196940, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315888, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 4, 0, 479244, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 479244, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333226')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1524759963631616.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.092849\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.492363\', 1, \'Factorizationmentotalkb\', \'248452.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001303\', 1, \'Fbscputimesec\', \'0.001303\', 1, \'Fbsmemorytotalkb\', \'204732.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'248452\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error =  62.366%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105236, 'I(1, 0, \'Frequency #5;  Interpolating frequency sweep\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 2.46GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 2.46GHz; S Matrix Error =  21.993%\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.73GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #6\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196336, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315312, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 4, 0, 462680, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 462680, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333228')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1644269978779648.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.090887\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.496255\', 1, \'Factorizationmentotalkb\', \'258269.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001144\', 1, \'Fbscputimesec\', \'0.001144\', 1, \'Fbsmemorytotalkb\', \'188720.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'258269\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 2.73GHz; S Matrix Error =   2.303%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105412, 'I(1, 0, \'Frequency #6;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.595GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #7\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196748, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315948, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 488052, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 488052, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333230')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2258139990720512.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.087935\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.501790\', 1, \'Factorizationmentotalkb\', \'266532.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001407\', 1, \'Fbscputimesec\', \'0.001407\', 1, \'Fbsmemorytotalkb\', \'213604.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'266532\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.595GHz; S Matrix Error =   0.329%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105416, 'I(1, 0, \'Frequency #7;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.23GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:00')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #8\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196672, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315244, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 4, 0, 471140, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 471140, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333233')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2806160100950016.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.092584\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.470005\', 1, \'Factorizationmentotalkb\', \'256598.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001397\', 1, \'Fbscputimesec\', \'0.001397\', 1, \'Fbsmemorytotalkb\', \'197732.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'256598\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.23GHz; S Matrix Error =   0.276%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105544, 'I(1, 0, \'Frequency #8;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.345GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #9\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196452, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315800, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 477264, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 477264, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333235')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2543189990834176.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.090204\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.537691\', 1, \'Factorizationmentotalkb\', \'253559.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001334\', 1, \'Fbscputimesec\', \'0.001334\', 1, \'Fbsmemorytotalkb\', \'202860.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'253559\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 2.345GHz; S Matrix Error =   0.091%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105952, 'I(1, 0, \'Frequency #9;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.75GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #10\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196788, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315180, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 464600, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 464600, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333237')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2789449993814016.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.087306\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.496377\', 1, \'Factorizationmentotalkb\', \'246844.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001292\', 1, \'Fbscputimesec\', \'0.001292\', 1, \'Fbsmemorytotalkb\', \'191016.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'246844\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 1.75GHz; S Matrix Error =   0.059%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106116, 'I(1, 0, \'Frequency #10;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.875GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #11\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 196552, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315352, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 5, 0, 472720, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 472720, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333239')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1996189935337472.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6501\', 1, \'Factornnz\', \'19620536\', 1, \'Factorestflops\', \'24209200000\', 1, \'Fbsestflops\', \'67514828\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.089183\', 1, \'Analysismemkb\', \'41868.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.503274\', 1, \'Factorizationmentotalkb\', \'248452.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001971\', 1, \'Fbscputimesec\', \'0.001971\', 1, \'Fbsmemorytotalkb\', \'198800.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'248452\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 12, Frequency: 1.875GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106176, 'I(1, 0, \'Frequency #11;  Interpolating frequency sweep\')', true, true)
				ProfileFootnote('I(1, 0, \'Interpolating sweep converged and is passive\')', 0)
				ProfileFootnote('I(1, 0, \'HFSS: Interpolating sweep\')', 0)
			$end 'ProfileGroup'
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Simulation Summary'
			$begin 'StartInfo'
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'97.8 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'0 Bytes\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:09\', 1, \'Average memory/process\', \'703 MB\', 1, \'Max memory/process\', \'703 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:23\', 1, \'Average memory/process\', \'467 MB\', 1, \'Max memory/process\', \'496 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 11350, false, 2, \'Max matrix size\', 74939, false, 1, \'Matrix bandwidth\', \'21.6\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'06/01/2026 20:00:40\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '06/01/2026 20:01:19')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:29')
			I(1, 'ComEngine Memory', '104 M')
		$end 'TotalInfo'
		GroupOptions=8
		TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Executing From\', \'C:\\\\Program Files\\\\ANSYS Inc\\\\v251\\\\AnsysEM\\\\HFSSCOMENGINE.exe\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='HPC'
			$begin 'StartInfo'
				I(1, 'Type', 'Manual')
				I(1, 'Distribution Types', 'Variations, Frequencies, Domain Solver, Transient Excitations, Mesh Assembly')
				I(1, 'MPI Vendor', 'Intel')
				I(1, 'MPI Version', '2021')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions(Memory=8)
			ProfileItem('Two Level', 0, 0, 0, 0, 0, 'I(1, 0, \'Disabled\')', false, true)
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'55.7 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 98.3 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 20:01:19')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:08')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 9'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 199620, 'I(2, 2, \'Tetrahedra\', 11350, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 315568, 'I(3, 2, \'Tetrahedra\', 11350, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'35.7 KB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 5, 0, 730660, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 74939, false, 3, \'Matrix bandwidth\', 21.6352, \'%5.1f\', 1, \'Disk\', \'295 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 730660, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'305 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 102936, 'I(1, 0, \'Adaptive Pass 9\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333281')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'74939\', 1, \'Matrixbw\', \'21.644400\', 1, \'Matrixnnz\', \'1622006\', 1, \'Rootdim\', \'1235\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1479699985334272.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6533\', 1, \'Factornnz\', \'19692472\', 1, \'Factorestflops\', \'24365600000\', 1, \'Fbsestflops\', \'67886262\', 1, \'Rootfactestflops\', \'627890811\', 1, \'Rootfbsestflops\', \'762612\', 1, \'Analysistimesec\', \'0.355745\', 1, \'Analysismemkb\', \'47064.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.540568\', 1, \'Factorizationmentotalkb\', \'538786.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.032534\', 1, \'Fbscputimesec\', \'0.032534\', 1, \'Fbsmemorytotalkb\', \'456948.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'538786\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 10'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 37772, 'I(2, 2, \'Tetrahedra\', 15354, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 206108, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 348880, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'346 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 1, 0, 8, 0, 898528, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 1, \'Disk\', \'367 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 898528, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'669 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103656, 'I(1, 0, \'Adaptive Pass 10\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1780333284')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1810830018478080.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8154\', 1, \'Factornnz\', \'26452738\', 1, \'Factorestflops\', \'38802300000\', 1, \'Fbsestflops\', \'91031639\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.446527\', 1, \'Analysismemkb\', \'58752.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.751816\', 1, \'Factorizationmentotalkb\', \'772740.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.050709\', 1, \'Fbscputimesec\', \'0.050709\', 1, \'Fbsmemorytotalkb\', \'601808.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'772740\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0146799, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '06/01/2026 20:01:27')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:21')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'HPC\', \'Enabled\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Solution - Sweep'
				$begin 'StartInfo'
					I(0, 'Interpolating HFSS Frequency Sweep')
					I(1, 'Time', '06/01/2026 20:01:27')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(1, 'Elapsed Time', '00:00:21')
				$end 'TotalInfo'
				GroupOptions=4
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'From 1GHz to 3GHz, 101 Frequencies\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 3GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #1; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202796, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349352, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 1, 0, 7, 0, 548792, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'367 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 548792, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333289')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2657850048380928.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.444686\', 1, \'Analysismemkb\', \'58228.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.661209\', 1, \'Factorizationmentotalkb\', \'321757.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002010\', 1, \'Fbscputimesec\', \'0.002010\', 1, \'Fbsmemorytotalkb\', \'251620.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'321757\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #2; Required Frequency\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202264, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 350024, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 564552, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 564552, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333291')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1707330030796800.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.113050\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.673021\', 1, \'Factorizationmentotalkb\', \'346789.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002004\', 1, \'Fbscputimesec\', \'0.002004\', 1, \'Fbsmemorytotalkb\', \'266552.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'346789\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 1, Frequency: 3GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 2, Frequency: 1GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #3\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202212, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349716, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 570504, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 570504, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333293')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2210559940362240.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.117110\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.650377\', 1, \'Factorizationmentotalkb\', \'368988.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001838\', 1, \'Fbscputimesec\', \'0.001838\', 1, \'Fbsmemorytotalkb\', \'272936.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'368988\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 142.481%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105280, 'I(1, 0, \'Frequency #3;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.5GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #4\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202376, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349528, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 7, 0, 553376, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 553376, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333295')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1541429939666944.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.112659\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.677691\', 1, \'Factorizationmentotalkb\', \'345344.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.003113\', 1, \'Fbscputimesec\', \'0.003113\', 1, \'Fbsmemorytotalkb\', \'257228.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'345344\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error =  86.540%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105456, 'I(1, 0, \'Frequency #4;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 1.25GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #5\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202632, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349452, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 557824, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 557824, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333297')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2487369911500800.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.113226\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.652047\', 1, \'Factorizationmentotalkb\', \'343564.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002056\', 1, \'Fbscputimesec\', \'0.002056\', 1, \'Fbsmemorytotalkb\', \'261296.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'343564\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error =  61.659%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105596, 'I(1, 0, \'Frequency #5;  Interpolating frequency sweep\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 2.4GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 2.4GHz; S Matrix Error =  22.016%\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.7GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #6\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202472, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349668, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 557720, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 557720, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333300')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1863089972576256.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.112543\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.655781\', 1, \'Factorizationmentotalkb\', \'331784.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002034\', 1, \'Fbscputimesec\', \'0.002034\', 1, \'Fbsmemorytotalkb\', \'261428.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'331784\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 2.7GHz; S Matrix Error =   2.634%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105732, 'I(1, 0, \'Frequency #6;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.55GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #7\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202192, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 1, 0, 348988, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 568680, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 568680, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333302')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2283739874852864.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.114333\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.676517\', 1, \'Factorizationmentotalkb\', \'342311.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002195\', 1, \'Fbscputimesec\', \'0.002195\', 1, \'Fbsmemorytotalkb\', \'272432.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'342311\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.55GHz; S Matrix Error =   0.412%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105772, 'I(1, 0, \'Frequency #7;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.85GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #8\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202252, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 1, 0, 348992, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 555640, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 555640, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333304')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2564460011061248.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.115251\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.658496\', 1, \'Factorizationmentotalkb\', \'335485.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001801\', 1, \'Fbscputimesec\', \'0.001801\', 1, \'Fbsmemorytotalkb\', \'259036.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'335485\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.85GHz; S Matrix Error =   0.162%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105824, 'I(1, 0, \'Frequency #8;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.2GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #9\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202400, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349256, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 564236, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 564236, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333306')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2081210020921344.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.115706\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.676110\', 1, \'Factorizationmentotalkb\', \'325869.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002126\', 1, \'Fbscputimesec\', \'0.002126\', 1, \'Fbsmemorytotalkb\', \'267032.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'325869\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 2.2GHz; S Matrix Error =   0.128%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106000, 'I(1, 0, \'Frequency #9;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.1GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #10\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 202544, 'I(2, 2, \'Tetrahedra\', 14187, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 349668, 'I(3, 2, \'Tetrahedra\', 14187, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 7, 0, 547296, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 93443, false, 3, \'Matrix bandwidth\', 21.6791, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 547296, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1780333308')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'93443\', 1, \'Matrixbw\', \'21.687300\', 1, \'Matrixnnz\', \'2026530\', 1, \'Rootdim\', \'1445\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2487140130750464.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'8123\', 1, \'Factornnz\', \'26388788\', 1, \'Factorestflops\', \'38680900000\', 1, \'Fbsestflops\', \'90885748\', 1, \'Rootfactestflops\', \'1005739572\', 1, \'Rootfbsestflops\', \'1044012\', 1, \'Analysistimesec\', \'0.113874\', 1, \'Analysismemkb\', \'52304.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.677713\', 1, \'Factorizationmentotalkb\', \'332159.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001468\', 1, \'Fbscputimesec\', \'0.001468\', 1, \'Fbsmemorytotalkb\', \'249740.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'332159\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 2.1GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106144, 'I(1, 0, \'Frequency #10;  Interpolating frequency sweep\')', true, true)
				ProfileFootnote('I(1, 0, \'Interpolating sweep converged and is passive\')', 0)
				ProfileFootnote('I(1, 0, \'HFSS: Interpolating sweep\')', 0)
			$end 'ProfileGroup'
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Simulation Summary'
			$begin 'StartInfo'
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'98.3 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'0 Bytes\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:08\', 1, \'Average memory/process\', \'877 MB\', 1, \'Max memory/process\', \'877 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:21\', 1, \'Average memory/process\', \'546 MB\', 1, \'Max memory/process\', \'557 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 14187, false, 2, \'Max matrix size\', 93443, false, 1, \'Matrix bandwidth\', \'21.7\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'06/01/2026 20:01:49\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
