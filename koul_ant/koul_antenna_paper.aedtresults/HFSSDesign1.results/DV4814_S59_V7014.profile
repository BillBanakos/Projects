$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '05/25/2026 22:12:09')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:49')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'63.5 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 98.5 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '05/25/2026 22:12:09')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:01')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
			ProfileItem('Mesh', 0, 0, 0, 0, 32864, 'I(2, 1, \'Type\', \'Phi\', 2, \'Tetrahedra\', 671, false)', true, true)
			ProfileItem('Post', 0, 0, 0, 0, 35300, 'I(2, 2, \'Tetrahedra\', 671, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Lambda Refine', 0, 0, 0, 0, 24564, 'I(2, 2, \'Tetrahedra\', 1705, false, 2, \'Cores\', 1, false)', true, true)
			ProfileItem('Simulation Setup', 0, 0, 0, 0, 174424, 'I(1, 1, \'Disk\', \'0 Bytes\')', true, true)
			ProfileItem('Port Adapt', 0, 0, 0, 0, 184632, 'I(2, 2, \'Tetrahedra\', 1547, false, 1, \'Disk\', \'33.5 KB\')', true, true)
			ProfileItem('Port Refine', 0, 0, 0, 0, 21932, 'I(2, 2, \'Tetrahedra\', 1793, false, 2, \'Cores\', 1, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '05/25/2026 22:12:11')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:26')
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
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 175940, 'I(2, 2, \'Tetrahedra\', 1620, false, 1, \'Disk\', \'4.31 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 199180, 'I(3, 2, \'Tetrahedra\', 1620, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'2.44 KB\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 249780, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 11471, false, 3, \'Matrix bandwidth\', 20.429, \'%5.1f\', 1, \'Disk\', \'48 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 249780, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'694 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103656, 'I(1, 0, \'Adaptive Pass 1\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736331')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'11471\', 1, \'Matrixbw\', \'20.457199\', 1, \'Matrixnnz\', \'234664\', 1, \'Rootdim\', \'419\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2648520037236736.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1024\', 1, \'Factornnz\', \'1624964\', 1, \'Factorestflops\', \'758994115\', 1, \'Fbsestflops\', \'5433351\', 1, \'Rootfactestflops\', \'24522179\', 1, \'Rootfbsestflops\', \'87780\', 1, \'Analysistimesec\', \'0.042762\', 1, \'Analysismemkb\', \'7732.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.048943\', 1, \'Factorizationmentotalkb\', \'54736.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.003692\', 1, \'Fbscputimesec\', \'0.003692\', 1, \'Fbsmemorytotalkb\', \'55700.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'55700\', 1, \'Maxdiskkb\', \'0\')', false, true)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 23424, 'I(2, 2, \'Tetrahedra\', 2281, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 177268, 'I(2, 2, \'Tetrahedra\', 2043, false, 1, \'Disk\', \'3.62 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 204228, 'I(3, 2, \'Tetrahedra\', 2043, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 269548, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 14291, false, 3, \'Matrix bandwidth\', 20.6496, \'%5.1f\', 1, \'Disk\', \'57.4 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 269548, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'182 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103912, 'I(1, 0, \'Adaptive Pass 2\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736334')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'14291\', 1, \'Matrixbw\', \'20.673300\', 1, \'Matrixnnz\', \'295442\', 1, \'Rootdim\', \'493\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2046140002336768.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1282\', 1, \'Factornnz\', \'2219592\', 1, \'Factorestflops\', \'1186622103\', 1, \'Fbsestflops\', \'7520338\', 1, \'Rootfactestflops\', \'39943779\', 1, \'Rootfbsestflops\', \'121524\', 1, \'Analysistimesec\', \'0.060354\', 1, \'Analysismemkb\', \'8600.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.069318\', 1, \'Factorizationmentotalkb\', \'72721.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.006530\', 1, \'Fbscputimesec\', \'0.006530\', 1, \'Fbsmemorytotalkb\', \'71924.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'72721\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.517255, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 24020, 'I(2, 2, \'Tetrahedra\', 2894, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 178348, 'I(2, 2, \'Tetrahedra\', 2582, false, 1, \'Disk\', \'3.21 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 210052, 'I(3, 2, \'Tetrahedra\', 2582, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 294760, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 17885, false, 3, \'Matrix bandwidth\', 20.8265, \'%5.1f\', 1, \'Disk\', \'71.5 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 294760, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'203 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 103932, 'I(1, 0, \'Adaptive Pass 3\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736336')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'17885\', 1, \'Matrixbw\', \'20.845900\', 1, \'Matrixnnz\', \'372829\', 1, \'Rootdim\', \'575\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'3081339997782016.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1569\', 1, \'Factornnz\', \'3063767\', 1, \'Factorestflops\', \'1933168726\', 1, \'Fbsestflops\', \'10325716\', 1, \'Rootfactestflops\', \'63373029\', 1, \'Rootfbsestflops\', \'165312\', 1, \'Analysistimesec\', \'0.071229\', 1, \'Analysismemkb\', \'10792.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.095497\', 1, \'Factorizationmentotalkb\', \'99735.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.008659\', 1, \'Fbscputimesec\', \'0.008659\', 1, \'Fbsmemorytotalkb\', \'93328.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'99735\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.3352, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 24788, 'I(2, 2, \'Tetrahedra\', 3672, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 180084, 'I(2, 2, \'Tetrahedra\', 3265, false, 1, \'Disk\', \'3.62 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 218688, 'I(3, 2, \'Tetrahedra\', 3265, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'2 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 324260, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 22441, false, 3, \'Matrix bandwidth\', 20.9717, \'%5.1f\', 1, \'Disk\', \'89.3 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 324260, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'230 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104008, 'I(1, 0, \'Adaptive Pass 4\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736339')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'22441\', 1, \'Matrixbw\', \'20.987101\', 1, \'Matrixnnz\', \'470971\', 1, \'Rootdim\', \'629\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'3022170112393216.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'1992\', 1, \'Factornnz\', \'4017807\', 1, \'Factorestflops\', \'2707840000\', 1, \'Fbsestflops\', \'13656368\', 1, \'Rootfactestflops\', \'82956220\', 1, \'Rootfbsestflops\', \'197820\', 1, \'Analysistimesec\', \'0.093474\', 1, \'Analysismemkb\', \'13700.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.111032\', 1, \'Factorizationmentotalkb\', \'128166.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.010797\', 1, \'Fbscputimesec\', \'0.010797\', 1, \'Fbsmemorytotalkb\', \'116860.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'128166\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0884145, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 26172, 'I(2, 2, \'Tetrahedra\', 4653, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 182328, 'I(2, 2, \'Tetrahedra\', 4138, false, 1, \'Disk\', \'3.62 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 229192, 'I(3, 2, \'Tetrahedra\', 4138, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 359712, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 28209, false, 3, \'Matrix bandwidth\', 21.1119, \'%5.1f\', 1, \'Disk\', \'112 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 359712, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'266 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104016, 'I(1, 0, \'Adaptive Pass 5\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736342')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'28209\', 1, \'Matrixbw\', \'21.125401\', 1, \'Matrixnnz\', \'595925\', 1, \'Rootdim\', \'717\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2559160021417984.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'2493\', 1, \'Factornnz\', \'5366821\', 1, \'Factorestflops\', \'3947450000\', 1, \'Fbsestflops\', \'18286878\', 1, \'Rootfactestflops\', \'122870890\', 1, \'Rootfbsestflops\', \'257044\', 1, \'Analysistimesec\', \'0.126351\', 1, \'Analysismemkb\', \'17160.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.151474\', 1, \'Factorizationmentotalkb\', \'161417.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.013104\', 1, \'Fbscputimesec\', \'0.013104\', 1, \'Fbsmemorytotalkb\', \'145144.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'161417\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0773162, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 27464, 'I(2, 2, \'Tetrahedra\', 5895, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 184968, 'I(2, 2, \'Tetrahedra\', 5251, false, 1, \'Disk\', \'3.59 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 243004, 'I(3, 2, \'Tetrahedra\', 5251, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 2, 0, 401912, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 35541, false, 3, \'Matrix bandwidth\', 21.2355, \'%5.1f\', 1, \'Disk\', \'140 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 401912, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'312 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104152, 'I(1, 0, \'Adaptive Pass 6\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736344')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'35541\', 1, \'Matrixbw\', \'21.247400\', 1, \'Matrixnnz\', \'755153\', 1, \'Rootdim\', \'757\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2348109891895296.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'3153\', 1, \'Factornnz\', \'6954407\', 1, \'Factorestflops\', \'5294780000\', 1, \'Fbsestflops\', \'23622042\', 1, \'Rootfactestflops\', \'144603459\', 1, \'Rootfbsestflops\', \'286524\', 1, \'Analysistimesec\', \'0.161498\', 1, \'Analysismemkb\', \'21516.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.193578\', 1, \'Factorizationmentotalkb\', \'209071.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.016730\', 1, \'Fbscputimesec\', \'0.016730\', 1, \'Fbsmemorytotalkb\', \'177624.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'209071\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0572176, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 29088, 'I(2, 2, \'Tetrahedra\', 7475, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 188708, 'I(2, 2, \'Tetrahedra\', 6718, false, 1, \'Disk\', \'3.98 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 260440, 'I(3, 2, \'Tetrahedra\', 6718, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 2, 0, 475024, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 45085, false, 3, \'Matrix bandwidth\', 21.3805, \'%5.1f\', 1, \'Disk\', \'178 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 475024, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'375 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104300, 'I(1, 0, \'Adaptive Pass 7\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736347')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'45085\', 1, \'Matrixbw\', \'21.389700\', 1, \'Matrixnnz\', \'964353\', 1, \'Rootdim\', \'883\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2172699937865728.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'3961\', 1, \'Factornnz\', \'9658491\', 1, \'Factorestflops\', \'8589730000\', 1, \'Fbsestflops\', \'32910121\', 1, \'Rootfactestflops\', \'229493282\', 1, \'Rootfbsestflops\', \'389844\', 1, \'Analysistimesec\', \'0.201650\', 1, \'Analysismemkb\', \'27916.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.253280\', 1, \'Factorizationmentotalkb\', \'287920.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.031805\', 1, \'Fbscputimesec\', \'0.031805\', 1, \'Fbsmemorytotalkb\', \'239112.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'287920\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0683555, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=1
				Name='Adaptive Pass 8'
				$begin 'StartInfo'
					I(1, 'Frequency', '2.4GHz')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 31452, 'I(2, 2, \'Tetrahedra\', 9492, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 193356, 'I(2, 2, \'Tetrahedra\', 8578, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 282760, 'I(3, 2, \'Tetrahedra\', 8578, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'346 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 4, 0, 580584, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 57251, false, 3, \'Matrix bandwidth\', 21.4703, \'%5.1f\', 1, \'Disk\', \'225 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 580584, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'450 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104404, 'I(1, 0, \'Adaptive Pass 8\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736350')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'57251\', 1, \'Matrixbw\', \'21.478901\', 1, \'Matrixnnz\', \'1229688\', 1, \'Rootdim\', \'1049\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2150350001799168.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'5044\', 1, \'Factornnz\', \'12812598\', 1, \'Factorestflops\', \'12681400000\', 1, \'Fbsestflops\', \'43890834\', 1, \'Rootfactestflops\', \'384779046\', 1, \'Rootfbsestflops\', \'550200\', 1, \'Analysistimesec\', \'0.267157\', 1, \'Analysismemkb\', \'35184.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.356411\', 1, \'Factorizationmentotalkb\', \'371705.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.030195\', 1, \'Fbscputimesec\', \'0.030195\', 1, \'Fbsmemorytotalkb\', \'328972.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'371705\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.02296, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 33944, 'I(2, 2, \'Tetrahedra\', 12069, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 198552, 'I(2, 2, \'Tetrahedra\', 10960, false, 1, \'Disk\', \'2.83 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 310820, 'I(3, 2, \'Tetrahedra\', 10960, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 0, 0, 5, 0, 693732, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 72791, false, 3, \'Matrix bandwidth\', 21.5538, \'%5.1f\', 1, \'Disk\', \'286 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 693732, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'550 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104440, 'I(1, 0, \'Adaptive Pass 9\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736353')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'72791\', 1, \'Matrixbw\', \'21.561399\', 1, \'Matrixnnz\', \'1569474\', 1, \'Rootdim\', \'1229\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1412930021097472.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'6389\', 1, \'Factornnz\', \'17308594\', 1, \'Factorestflops\', \'18979800000\', 1, \'Fbsestflops\', \'59342831\', 1, \'Rootfactestflops\', \'618783892\', 1, \'Rootfbsestflops\', \'755220\', 1, \'Analysistimesec\', \'0.339934\', 1, \'Analysismemkb\', \'45536.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.507544\', 1, \'Factorizationmentotalkb\', \'493112.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.036588\', 1, \'Fbscputimesec\', \'0.036588\', 1, \'Fbsmemorytotalkb\', \'423088.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'493112\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.0266871, \'%.5f\')', 0)
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
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 36964, 'I(2, 2, \'Tetrahedra\', 14934, false, 2, \'Cores\', 1, false)', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('Simulation Setup ', 0, 0, 0, 0, 204584, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'3.21 KB\')', true, true)
				ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342236, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'347 Bytes\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
				ProfileItem('Matrix Solve', 1, 0, 8, 0, 815808, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'353 KB\')', true, true)
				ProfileItem('Field Recovery', 0, 0, 0, 0, 815808, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'626 KB\')', true, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104508, 'I(1, 0, \'Adaptive Pass 10\')', true, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='APIPms'
					$begin 'StartInfo'
						I(1, 'Timesinceepock', '1779736357')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, ' ')
					$end 'TotalInfo'
					GroupOptions=16
					TaskDataOptions(Memory=8)
					ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
					ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'3130800035856384.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
					ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.433935\', 1, \'Analysismemkb\', \'56348.000000\')', false, true)
					ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.712225\', 1, \'Factorizationmentotalkb\', \'635772.000000\')', false, true)
					ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.047449\', 1, \'Fbscputimesec\', \'0.047449\', 1, \'Fbsmemorytotalkb\', \'523712.000000\')', false, true)
					ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'635772\', 1, \'Maxdiskkb\', \'0\')', false, true)
				$end 'ProfileGroup'
				ProfileFootnote('I(1, 3, \'Max Mag. Delta S\', 0.017411, \'%.5f\')', 0)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '05/25/2026 22:12:37')
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
					I(1, 'Time', '05/25/2026 22:12:37')
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
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201060, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342356, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 1, 0, 7, 0, 533468, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'353 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 533468, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736359')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'3157840076210176.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.440657\', 1, \'Analysismemkb\', \'55932.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.637261\', 1, \'Factorizationmentotalkb\', \'281212.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002341\', 1, \'Fbscputimesec\', \'0.002341\', 1, \'Fbsmemorytotalkb\', \'241020.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'281212\', 1, \'Maxdiskkb\', \'0\')', false, true)
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
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201180, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342248, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'2 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 529004, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 529004, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736361')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2141860059414528.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111526\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.673191\', 1, \'Factorizationmentotalkb\', \'292800.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002458\', 1, \'Fbscputimesec\', \'0.002458\', 1, \'Fbsmemorytotalkb\', \'236708.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'292800\', 1, \'Maxdiskkb\', \'0\')', false, true)
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
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201188, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342580, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'9 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 525368, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 525368, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736363')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2676259888824320.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111106\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.655422\', 1, \'Factorizationmentotalkb\', \'301193.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002110\', 1, \'Fbscputimesec\', \'0.002110\', 1, \'Fbsmemorytotalkb\', \'232756.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'301193\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 174.481%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106632, 'I(1, 0, \'Frequency #3;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 200924, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 1, 0, 341536, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 514672, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 514672, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736365')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2134819936927744.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.106768\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.637586\', 1, \'Factorizationmentotalkb\', \'283925.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002177\', 1, \'Fbscputimesec\', \'0.002177\', 1, \'Fbsmemorytotalkb\', \'222992.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'283925\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error = 107.499%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106820, 'I(1, 0, \'Frequency #4;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201372, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342096, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'10 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 539048, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 539048, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736368')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1458970057244672.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111153\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.646577\', 1, \'Factorizationmentotalkb\', \'292800.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001928\', 1, \'Fbscputimesec\', \'0.001928\', 1, \'Fbsmemorytotalkb\', \'246756.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'292800\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error = 119.717%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 106940, 'I(1, 0, \'Frequency #5;  Interpolating frequency sweep\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 2.4GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 2.4GHz; S Matrix Error =  74.585%\')', false, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #6\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201060, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342196, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'4 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 532176, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 532176, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736370')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2177989995397120.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.109479\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.659505\', 1, \'Factorizationmentotalkb\', \'293735.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002388\', 1, \'Fbscputimesec\', \'0.002388\', 1, \'Fbsmemorytotalkb\', \'239868.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'293735\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 2.2GHz; S Matrix Error =  22.047%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107160, 'I(1, 0, \'Frequency #6;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #7\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 200952, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 341640, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 522752, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 522752, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736372')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1477260041256960.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.108656\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.616374\', 1, \'Factorizationmentotalkb\', \'289428.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002113\', 1, \'Fbscputimesec\', \'0.002113\', 1, \'Fbsmemorytotalkb\', \'230968.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'289428\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.7GHz; S Matrix Error =  14.224%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107164, 'I(1, 0, \'Frequency #7;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #8\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201028, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342764, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'3 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 532320, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 532320, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736374')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1529189953961984.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.105979\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.597449\', 1, \'Factorizationmentotalkb\', \'284321.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.001975\', 1, \'Fbscputimesec\', \'0.001975\', 1, \'Fbsmemorytotalkb\', \'239472.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'284321\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.55GHz; S Matrix Error =   2.272%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107184, 'I(1, 0, \'Frequency #8;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #9\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201052, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342216, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 525444, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 525444, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736376')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2881539998220288.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.110857\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.655629\', 1, \'Factorizationmentotalkb\', \'285001.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.003426\', 1, \'Fbscputimesec\', \'0.003426\', 1, \'Fbsmemorytotalkb\', \'233124.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'285001\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 1.75GHz; S Matrix Error =   0.237%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107236, 'I(1, 0, \'Frequency #9;  Interpolating frequency sweep\')', true, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #10\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 200876, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 341744, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'7 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 518660, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 518660, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.79 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736378')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2414289969217536.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111812\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.665876\', 1, \'Factorizationmentotalkb\', \'278262.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002562\', 1, \'Fbscputimesec\', \'0.002562\', 1, \'Fbsmemorytotalkb\', \'226784.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'278262\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 1.875GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 107272, 'I(1, 0, \'Frequency #10;  Interpolating frequency sweep\')', true, true)
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
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'98.5 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:01\', 1, \'Total Memory\', \'215 MB\')', false, true)
			ProfileItem('Adaptive Meshing', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:26\', 1, \'Average memory/process\', \'797 MB\', 1, \'Max memory/process\', \'797 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:21\', 1, \'Average memory/process\', \'515 MB\', 1, \'Max memory/process\', \'526 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 13584, false, 2, \'Max matrix size\', 89965, false, 1, \'Matrix bandwidth\', \'21.6\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'05/25/2026 22:12:59\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '05/25/2026 22:18:49')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:02')
			I(1, 'ComEngine Memory', '102 M')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'63.5 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 97 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Adaptive Passes converged\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '05/25/2026 22:18:49')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:02')
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
					I(1, 'Time', '05/25/2026 22:18:49')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(1, 'Elapsed Time', '00:00:02')
				$end 'TotalInfo'
				GroupOptions=4
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'From 1GHz to 3GHz, 101 Frequencies\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 1, Frequency: 3GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 2, Frequency: 1GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 174.481%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error = 107.499%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error = 119.717%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 1.75GHz; S Matrix Error =  63.924%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 1.875GHz; S Matrix Error =  23.504%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.4GHz; S Matrix Error =  42.251%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.55GHz; S Matrix Error =   5.693%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 2.7GHz; S Matrix Error =   0.825%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 2.2GHz; S Matrix Error =   0.059%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using previously solved data. Additional simulations must be performed to correct interpolating sweep convergence or passivity\')', false, true)
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
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #1\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 200928, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 341088, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'3.7 KB\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 514464, 'I(6, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 0, \'s-matrix only solve\', 1, \'Disk\', \'2.43 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 514464, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'2.8 KB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736731')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2305799900626944.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111074\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.629531\', 1, \'Factorizationmentotalkb\', \'292383.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'partial_dense\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'true\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.002098\', 1, \'Fbscputimesec\', \'0.002098\', 1, \'Fbsmemorytotalkb\', \'223256.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'292383\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 12, Frequency: 2.1GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104536, 'I(1, 0, \'Frequency #12;  Interpolating frequency sweep\')', true, true)
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
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'97 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'0 Bytes\')', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:02\', 1, \'Average memory/process\', \'502 MB\', 1, \'Max memory/process\', \'502 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 13584, false, 2, \'Max matrix size\', 89965, false, 1, \'Matrix bandwidth\', \'21.6\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'05/25/2026 22:18:51\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=1
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '05/25/2026 22:20:11')
			I(1, 'Host', 'PC-BILL')
			I(1, 'Processor', '20')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'HFSS Version 2025.1.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:11')
			I(1, 'ComEngine Memory', '103 M')
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
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(6, 1, \'Name\', \'PC-BILL\', 1, \'Memory\', \'15.8 GB\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 12, false, 1, \'Free Disk Space\', \'63.5 GB\')', false, true)
		$end 'ProfileGroup'
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Allow off core\', \'True\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Solution Basis Order\', \'1\')', false, true)
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(1, 0, \'Elapsed time : 00:00:00 , HFSS ComEngine Memory : 97.3 M\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Perform full validations with standard port validations\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Adaptive Passes converged\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=1
			Name='Frequency Sweep'
			$begin 'StartInfo'
				I(1, 'Time', '05/25/2026 22:20:11')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:11')
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
					I(1, 'Time', '05/25/2026 22:20:11')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(1, 'Elapsed Time', '00:00:11')
				$end 'TotalInfo'
				GroupOptions=4
				TaskDataOptions('CPU Time'=8, Memory=8, 'Real Time'=8)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'From 1GHz to 3GHz, 101 Frequencies\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using previously solved data. Additional simulations must be performed to correct interpolating sweep convergence or passivity\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 3GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 1GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 1, Frequency: 3GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 2, Frequency: 1GHz; Additional basis points are needed before the interpolation error can be computed.\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 2GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 3, Frequency: 2GHz; S Matrix Error = 174.481%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 1.5GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 4, Frequency: 1.5GHz; S Matrix Error = 107.499%\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 1.25GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 5, Frequency: 1.25GHz; S Matrix Error = 119.717%\')', false, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.5GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #1\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201232, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342652, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'3.69 KB\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 797080, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'2.42 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 797080, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'3.55 MB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736813')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2562299910946816.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111711\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.675360\', 1, \'Factorizationmentotalkb\', \'638154.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.044615\', 1, \'Fbscputimesec\', \'0.044615\', 1, \'Fbsmemorytotalkb\', \'504308.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'638154\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 6, Frequency: 2.5GHz; S Matrix Error =  63.920%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104632, 'I(1, 0, \'Frequency #1;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.25GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #2\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201316, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342524, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'4 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 7, 0, 804476, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 804476, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'3.55 MB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736815')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2084270050902016.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111151\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.704798\', 1, \'Factorizationmentotalkb\', \'653124.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.045619\', 1, \'Fbscputimesec\', \'0.045619\', 1, \'Fbsmemorytotalkb\', \'511736.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'653124\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 7, Frequency: 2.25GHz; S Matrix Error =  13.772%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104828, 'I(1, 0, \'Frequency #2;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.75GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #3\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201260, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342084, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 7, 0, 810416, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 810416, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'3.55 MB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736817')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2295429970526208.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.110186\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.733182\', 1, \'Factorizationmentotalkb\', \'635772.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.043109\', 1, \'Fbscputimesec\', \'0.043109\', 1, \'Fbsmemorytotalkb\', \'518168.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'635772\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 8, Frequency: 2.75GHz; S Matrix Error =   8.803%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104888, 'I(1, 0, \'Frequency #3;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.625GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #4\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201216, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 342256, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 797720, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 797720, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'3.55 MB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736819')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'1873489967448064.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.111039\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.672053\', 1, \'Factorizationmentotalkb\', \'641729.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.045044\', 1, \'Fbscputimesec\', \'0.045044\', 1, \'Fbsmemorytotalkb\', \'505216.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'641729\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 9, Frequency: 2.625GHz; S Matrix Error =   3.093%\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 104936, 'I(1, 0, \'Frequency #4;  Interpolating frequency sweep\')', true, true)
				ProfileItem(' ', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				$begin 'ProfileGroup'
					MajorVer=2025
					MinorVer=1
					Name='Frequency - 2.125GHz'
					$begin 'StartInfo'
						I(0, 'PC-BILL')
					$end 'StartInfo'
					$begin 'TotalInfo'
						I(0, 'Elapsed time : 00:00:01')
					$end 'TotalInfo'
					GroupOptions=0
					TaskDataOptions('CPU Time'=8, 'Real Time'=8)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'HFSS: Single Frequency Solve Group #5\')', false, true)
					ProfileItem('Simulation Setup ', 0, 0, 0, 0, 201144, 'I(2, 2, \'Tetrahedra\', 13584, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('Matrix Assembly', 0, 0, 0, 0, 341784, 'I(3, 2, \'Tetrahedra\', 13584, false, 2, \'Lumped ports\', 1, false, 1, \'Disk\', \'0 Bytes\')', true, true)
					ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Using Direct Solver due to the matrix size being small\\\
\')', false, true)
					ProfileItem('Matrix Solve', 0, 0, 6, 0, 796640, 'I(5, 1, \'Type\', \'DCS\', 2, \'Cores\', 12, false, 2, \'Matrix size\', 89965, false, 3, \'Matrix bandwidth\', 21.6016, \'%5.1f\', 1, \'Disk\', \'1.61 KB\')', true, true)
					ProfileItem('Field Recovery', 0, 0, 0, 0, 796640, 'I(2, 2, \'Excitations\', 1, false, 1, \'Disk\', \'3.55 MB\')', true, true)
					$begin 'ProfileGroup'
						MajorVer=2025
						MinorVer=1
						Name='APIPms1'
						$begin 'StartInfo'
							I(1, 'Timesinceepock', '1779736821')
						$end 'StartInfo'
						$begin 'TotalInfo'
							I(0, ' ')
						$end 'TotalInfo'
						GroupOptions=16
						TaskDataOptions('CPU Time'=8, 'Real Time'=8)
						ProfileItem('solverinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Solvertype\', \'shared_memory\', 1, \'Precision\', \'double\', 1, \'Solversymmetry\', \'complex_sym\', 1, \'Matrixdim\', \'89965\', 1, \'Matrixbw\', \'21.608801\', 1, \'Matrixnnz\', \'1944037\', 1, \'Rootdim\', \'1383\', 1, \'Mathtype\', \'mkl\', 1, \'Mpitasks\', \'1\', 1, \'Threadspertasks\', \'0\')', false, true)
						ProfileItem('sysinfo', 0, 0, 0, 0, 0, 'I(12, 1, \'Os\', \'win\', 1, \'Cpuid\', \'13th Gen Intel(R) Core(TM) i5-13500\', 1, \'CpuPhysicCores\', \'14\', 1, \'CpuLogicCores\', \'20\', 1, \'Cpufreqkhz\', \'2102680025563136.000000\', 1, \'Cpucachelinesizebytes\', \'64\', 1, \'Cpuestlastlevelcachesizemb\', \'24.000000\', 1, \'Cpuestgflops\', \'448.000000\', 1, \'Memorybwestkbps\', \'76.800003\', 1, \'Numanodes\', \'1\', 1, \'Virtualmemkb\', \'137439000000.000000\', 1, \'Pagesizekb\', \'4096\')', false, true)
						ProfileItem('analysisinfo', 0, 0, 0, 0, 0, 'I(9, 1, \'Analysisstatus\', \'valid\', 1, \'Numsupernodes\', \'7882\', 1, \'Factornnz\', \'22880649\', 1, \'Factorestflops\', \'29477700000\', 1, \'Fbsestflops\', \'78471026\', 1, \'Rootfactestflops\', \'881756745\', 1, \'Rootfbsestflops\', \'956344\', 1, \'Analysistimesec\', \'0.106667\', 1, \'Analysismemkb\', \'50190.000000\')', false, true)
						ProfileItem('factorinfo', 0, 0, 0, 0, 0, 'I(4, 1, \'Fatorizationstatus\', \'valid\', 1, \'Factorizationnumcores\', \'12\', 1, \'Factorizationtimesec\', \'0.652255\', 1, \'Factorizationmentotalkb\', \'654415.000000\')', false, true)
						ProfileItem('fbsinfo', 0, 0, 0, 0, 0, 'I(10, 1, \'Fbstatus\', \'valid\', 1, \'Fbstype\', \'fullsolve\', 1, \'Fbsmt\', \'false\', 1, \'Fbsmrhs\', \'false\', 1, \'Fbsnumcores\', \'12\', 1, \'Fbsnumsolvestotal\', \'1\', 1, \'Fbsnumsolves\', \'1\', 1, \'Fbsavgsolvetime1solvesec\', \'0.044269\', 1, \'Fbscputimesec\', \'0.044269\', 1, \'Fbsmemorytotalkb\', \'505848.000000\')', false, true)
						ProfileItem('solverprofile', 0, 0, 0, 0, 0, 'I(2, 1, \'Maxmemkb\', \'654415\', 1, \'Maxdiskkb\', \'0\')', false, true)
					$end 'ProfileGroup'
				$end 'ProfileGroup'
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 10, Frequency: 2.125GHz; S Matrix Error =   0.446%; Secondary solver criterion is not converged\')', false, true)
				ProfileItem('Data Transfer', 0, 0, 0, 0, 105032, 'I(1, 0, \'Frequency #5;  Interpolating frequency sweep\')', true, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Frequency: 1.75GHz has already been solved\')', false, true)
				ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'Basis Element # 11, Frequency: 1.75GHz; Scattering matrix quantities converged; Passive within tolerance\')', false, true)
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
			ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'97.3 MB\')', false, true)
			ProfileItem('Initial Meshing', 0, 0, 0, 0, 0, 'I(2, 1, \'Elapsed Time\', \'00:00:00\', 1, \'Total Memory\', \'0 Bytes\')', false, true)
			ProfileItem('Frequency Sweep', 0, 0, 0, 0, 0, 'I(5, 1, \'Elapsed Time\', \'00:00:11\', 1, \'Average memory/process\', \'782 MB\', 1, \'Max memory/process\', \'791 MB\', 2, \'Max number of processes/frequency\', 1, false, 2, \'Total number of cores\', 12, false)', false, true)
			ProfileFootnote('I(3, 2, \'Max solved tets\', 13584, false, 2, \'Max matrix size\', 89965, false, 1, \'Matrix bandwidth\', \'21.6\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'05/25/2026 22:20:22\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
