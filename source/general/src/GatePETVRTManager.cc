/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GatePETVRTManager.hh"
#include "GatePETVRTSettings.hh"
#include "GateMessageManager.hh"

GatePETVRTManager* GatePETVRTManager::m_sInstance=NULL;

// NOT thread-safe!!
GatePETVRTManager* GatePETVRTManager::GetInstance()
{
	static GatePETVRTManager inst;
	if ( m_sInstance == NULL )
	{
		m_sInstance=&inst;
	}
	return m_sInstance;
}

GatePETVRTManager::GatePETVRTManager() :pGatePETVRTSettings ( NULL )
{
	m_nDeletePETVRTSettings=false;
}

GatePETVRTManager::~GatePETVRTManager()
{
	if (m_nDeletePETVRTSettings)
		delete pGatePETVRTSettings;
}

GateMaterialTableToProductionCutsTable* GatePETVRTManager::GetMaterialTableToProductionCutsTable()
{
	return &m_oMPTable;
}

void GatePETVRTManager::DeletePETVRTSettings ()
{
	if (m_nDeletePETVRTSettings) delete pGatePETVRTSettings;
	else G4cout << "GatePETVRTManager::DeletePETVRTSettings: Try to delete, but GatePETVRTSettings not deletable!\n";
	pGatePETVRTSettings=NULL;
}

void GatePETVRTManager::RegisterPETVRTSettings ( GatePETVRTSettings* s, bool del )
{
	if ((pGatePETVRTSettings!=NULL)&& m_nDeletePETVRTSettings )
	{
		G4cout << "GatePETVRTManager::SetPETVRTSettings: PETVRTSettings already registered. Delete first. Do nothing!\n";
	}
	else
	pGatePETVRTSettings=s;
	m_nDeletePETVRTSettings=del;
}

GatePETVRTSettings* GatePETVRTManager::GetOrCreatePETVRTSettings()
{
	if (pGatePETVRTSettings==NULL)
	{
		pGatePETVRTSettings=new GatePETVRTSettings();
		m_nDeletePETVRTSettings=true;
	}
	return pGatePETVRTSettings;
}
