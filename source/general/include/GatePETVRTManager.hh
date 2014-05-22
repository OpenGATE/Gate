/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GatePETVRTManager_hh
#define GatePETVRTManager_hh
/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/
#include "GateMaterialTableToProductionCutsTable.hh"
class GatePETVRTSettings;

// This implementation is NOT thread-safe!
class GatePETVRTManager
{
	public:
		static GatePETVRTManager* GetInstance();

		GateMaterialTableToProductionCutsTable* GetMaterialTableToProductionCutsTable();
		void RegisterPETVRTSettings ( GatePETVRTSettings*, bool deleteWithManager );
		GatePETVRTSettings* GetOrCreatePETVRTSettings();
		void DeletePETVRTSettings();
	protected:
		GatePETVRTManager();
		~GatePETVRTManager();
	private:
		GateMaterialTableToProductionCutsTable m_oMPTable;
		static GatePETVRTManager* m_sInstance;
		GatePETVRTSettings* pGatePETVRTSettings;
		bool m_nDeletePETVRTSettings;
};


#endif
