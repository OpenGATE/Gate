

/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateSpatialResolution
  \brief  Digitizer Module for simulating a Gaussian blurring on the position.

  Includes functionalities from:
  	  GateSpblurring
	  GateCC3DlocalSpblurring
	  GateDoIModels

  Previous authors: Steven.Staelens@rug.ac.be(?), AE

  - modified by Adrien Paillet 11/2022
  	  This blurring has been validated up to a given FWHM of 10mm.
  	  At higher FWHM, the number of "relocated" digis is no longer negligible. The blurring effect is then so compensated that resolution will improve compared to lower values of FWHM.
-modified by Radia Oudihat 06/2024
        Added support for 1D FWHM distribution for X and Y, and applied Gaussian blurring.
        Implemented logic to determine standard deviations (stddevX, stddevY) based on defined 1D and 2D FWHM distributions for the X and Y axes.
-modified by Marc Granado-Gonzalez 2025
		- Added 2D FWHM distribution for X, Y and Z, and applied Gaussian blurring.
		Implemented logic to determine standard deviations (stddevX, stddevY, stddevZ) based on defined 2D FWHM distributions for the X, Y and Z axes.
		- Added option to choose the axis pair for 2D distributions (nameAxis): "XZ" or "YZ" (default "YZ").
		- Added Truncated Gaussian option for confined and non-confined cases.
		*/


#include "GateSpatialResolution.hh"
#include "GateSpatialResolutionMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateObjectStore.hh"
#include "GateConstants.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "GateVDistribution.hh"
#include "GateDistributionTruncatedGaussian.hh"




GateSpatialResolution::GateSpatialResolution(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_fwhm(0),
   m_fwhmX(0),
   m_fwhmY(0),
   m_fwhmZ(0),
	 m_fwhmXDistrib(0),
	 m_fwhmYDistrib(0),
	 m_fwhmZDistrib(0),
 	m_nameAxis("YZ"),
	m_fwhmXDistrib2D(0),
	m_fwhmYDistrib2D(0),
	m_fwhmZDistrib2D(0),
   m_IsConfined(true),
   m_UseTruncatedGaussian(false),
   m_Navigator(0),
   m_Touchable(0),
   m_systemDepth(-1),
   m_outputDigi(0),
   m_IsFirstEntrance(1),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateSpatialResolutionMessenger(this);
}


GateSpatialResolution::~GateSpatialResolution()
{
  delete m_Messenger;

}
void GateSpatialResolution::SetSpatialResolutionParameters() {
    // Check FWHM parameters
	if (m_fwhm != 0 && (m_fwhmX != 0 || m_fwhmY != 0 || m_fwhmZ != 0 || m_fwhmXDistrib2D != 0 || m_fwhmYDistrib2D != 0 || m_fwhmZDistrib2D != 0 || m_fwhmXDistrib !=0 || m_fwhmYDistrib !=0 || m_fwhmZDistrib !=0 )) {
        G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set a unique FWHM for all 3 axes OR set FWHM for X, Y, Z individually." << G4endl;
        abort();
    }

	if (m_fwhmXDistrib2D || m_fwhmYDistrib2D || m_fwhmZDistrib2D)
	{
		// If a per-axis 2D distribution is provided for an axis, it is
		// ambiguous to also provide a scalar FWHM or a 1D distribution for
		// the same axis. Check per-axis instead of using nameAxis combinatorics.
		if (m_fwhmXDistrib2D && (m_fwhmX != 0 || m_fwhmXDistrib != 0)) {
			G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for X OR set FWHM distribution for X." << G4endl;
			abort();
		}
		if (m_fwhmYDistrib2D && (m_fwhmY != 0 || m_fwhmYDistrib != 0)) {
			G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for Y OR set FWHM distribution for Y." << G4endl;
			abort();
		}
		if (m_fwhmZDistrib2D && (m_fwhmZ != 0 || m_fwhmZDistrib != 0)) {
			G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for Z OR set FWHM distribution for Z." << G4endl;
			abort();
		}
	}

	if (m_fwhmX != 0 && m_fwhmXDistrib !=0){
    	G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for X OR set FWHM for Z distribution." << G4endl;
    	abort();
   	}
    if (m_fwhmY != 0 && m_fwhmYDistrib !=0){
    	G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for Y OR set FWHM for Z distribution." << G4endl;
    	abort();
   	}
    if (m_fwhmZ != 0 && m_fwhmZDistrib !=0){
    	G4cout << "***ERROR*** Spatial Resolution is ambiguous: you can set FWHM for Z OR set FWHM for Z distribution." << G4endl;
    	abort();
   	}

}



void GateSpatialResolution::Digitize(){

 
	GateVSystem* m_system =  ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();

	  if (m_IsFirstEntrance) {
		  SetSpatialResolutionParameters();
			if (!m_system->CheckIfEnoughLevelsAreDefined())
			{
				 GateError( " *** ERROR*** GateSpatialResolution::Digitize. Not all defined geometry levels has their mother levels defined."
						 "(Ex.: for cylindricalPET, the levels are: rsector, module, submodule, crystal). If you have defined submodule, you have to have resector and module defined as well."
						 "Please, add them to your geometry macro in /gate/systems/cylindricalPET/XXX/attach    YYY. Abort.\n");
			}

	        m_IsFirstEntrance = false;
	    }


	G4double fwhmX;
	G4double fwhmY;
	G4double fwhmZ;

	if (m_fwhmX == 0 && m_fwhmY == 0 && m_fwhmZ == 0)
	{
	    fwhmX = m_fwhm;
	    fwhmY = m_fwhm;
	    fwhmZ = m_fwhm;
	}
	else
	{
	    fwhmX = m_fwhmX;
	    fwhmY = m_fwhmY;
	    fwhmZ = m_fwhmZ;
	}




	if (m_system==NULL) G4Exception( "GateSpatialResolution::Digitize", "Digitize", FatalException,
				 "Failed to get the system corresponding to that digitizer. Abort.\n");


	m_systemDepth = m_system->GetTreeDepth();


	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;


	/*	if(!m_IsConfined && !m_Navigator)
	  {
	    //Getting world Volume
	    // Do not use from TransportationManager as it is not recommended
	    G4Navigator *navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
	    G4VPhysicalVolume *WorldVolume = navigator->GetWorldVolume();
	    m_Navigator = new G4Navigator();
	    m_Navigator->SetWorldVolume(WorldVolume);
	  }
	*/

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  m_outputDigi = new GateDigi(*inputDigi);


		  G4ThreeVector P = inputDigi->GetVolumeID().MoveToBottomVolumeFrame(inputDigi->GetGlobalPos()); //TC

		  G4double Px = P.x();
		  G4double Py = P.y();
		  G4double Pz = P.z();
		  G4double stddevX = 0., stddevY = 0., stddevZ = 0.;

		  // Use the configured axis pair (m_nameAxis) to evaluate Value2D.
		  // Allowed pairs for PET: "XZ" or "YZ" (default "YZ").
		  if (m_fwhmXDistrib2D || m_fwhmYDistrib2D || m_fwhmZDistrib2D) {
			  if (m_nameAxis == "XZ") {
				  if (m_fwhmXDistrib2D) stddevX = m_fwhmXDistrib2D->Value2D(P.x() * mm, P.z() * mm);
				  if (m_fwhmYDistrib2D) stddevY = m_fwhmYDistrib2D->Value2D(P.x() * mm, P.z() * mm);
				  if (m_fwhmZDistrib2D) stddevZ = m_fwhmZDistrib2D->Value2D(P.x() * mm, P.z() * mm);
			  } else { // YZ
				  if (m_fwhmXDistrib2D) stddevX = m_fwhmXDistrib2D->Value2D(P.y() * mm, P.z() * mm);
				  if (m_fwhmYDistrib2D) stddevY = m_fwhmYDistrib2D->Value2D(P.y() * mm, P.z() * mm);
				  if (m_fwhmZDistrib2D) stddevZ = m_fwhmZDistrib2D->Value2D(P.y() * mm, P.z() * mm);
			  }
		  }
		 else {

			 if (m_fwhmXDistrib) stddevX = m_fwhmXDistrib->Value(P.x() * mm);
			 else if (fwhmX) stddevX = fwhmX / GateConstants::fwhm_to_sigma;

			 if (m_fwhmYDistrib) stddevY = m_fwhmYDistrib->Value(P.y() * mm);
			 else if (fwhmY) stddevY = fwhmY / GateConstants::fwhm_to_sigma;

			 if (m_fwhmZDistrib) stddevZ = m_fwhmZDistrib->Value(P.z() * mm);
			 else if (fwhmZ) stddevZ = fwhmZ / GateConstants::fwhm_to_sigma;

		  }



			  G4double PxNew ;
			  G4double PyNew ;
			  G4double PzNew ;

// store the computed stddevs into the digi for later ROOT output
		  m_outputDigi->SetSpatialRes2DStdDevX(stddevX);
		  m_outputDigi->SetSpatialRes2DStdDevY(stddevY);
		  m_outputDigi->SetSpatialRes2DStdDevZ(stddevZ);



		  if (m_IsConfined)
		  {
			  //set the position on the border of the crystal
			  //no need to update volume ID
			inputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, Xmin, Xmax);
			inputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, Ymin, Ymax);
			inputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, Zmin, Zmax);

			if (m_UseTruncatedGaussian)
					  {

						 PxNew = GateDistributionTruncatedGaussian::shootRandom(Px,stddevX,Xmin, Xmax);
						 PyNew = GateDistributionTruncatedGaussian::shootRandom(Py,stddevY,Ymin, Ymax);
						 PzNew = GateDistributionTruncatedGaussian::shootRandom(Pz,stddevZ,Zmin, Zmax);
					  }
			else{

				PxNew = G4RandGauss::shoot(Px,stddevX);
				PyNew = G4RandGauss::shoot(Py,stddevY);
				PzNew = G4RandGauss::shoot(Pz,stddevZ);

			if(PxNew<Xmin) PxNew=Xmin;
			if(PyNew<Ymin) PyNew=Ymin;
			if(PzNew<Zmin) PzNew=Zmin;
			if(PxNew>Xmax) PxNew=Xmax;
			if(PyNew>Ymax) PyNew=Ymax;
			if(PzNew>Zmax) PzNew=Zmax;
			}


			m_outputDigi->SetLocalPos(G4ThreeVector(PxNew,PyNew,PzNew)); //TC
			//G4cout<<G4ThreeVector(PxNew,PyNew,PzNew)<<G4endl;
			//G4cout<<m_outputDigi->GetVolumeID().MoveToAncestorVolumeFrame(m_outputDigi->GetLocalPos())<<G4endl;
			m_outputDigi->SetGlobalPos(m_outputDigi->GetVolumeID().MoveToAncestorVolumeFrame(m_outputDigi->GetLocalPos())); //TC
			//TC
			//outputPulse->SetGlobalPos(G4ThreeVector(PxNew,PyNew,PzNew));
			m_OutputDigiCollection->insert(m_outputDigi);
			//G4cout<<"PxNew"<<PxNew<<Gateendl;
		  }
		  else
		  {
			//Not confined:
			//Update volume IDs and new locations inside crystal
			  // TODO Test properly and maybe extent to more general cases
			  inputDigi->GetVolumeID().GetCreator(m_systemDepth-1)->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, Xmin, Xmax);
  			  inputDigi->GetVolumeID().GetCreator(m_systemDepth-1)->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, Ymin, Ymax);
  			  inputDigi->GetVolumeID().GetCreator(m_systemDepth-1)->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, Zmin, Zmax);


				if (m_UseTruncatedGaussian)
						  {

							 PxNew = GateDistributionTruncatedGaussian::shootRandom(Px,stddevX,Xmin, Xmax);
							 PyNew = GateDistributionTruncatedGaussian::shootRandom(Py,stddevY,Ymin, Ymax);
							 PzNew = GateDistributionTruncatedGaussian::shootRandom(Pz,stddevZ,Zmin, Zmax);
						  }
				else{

					PxNew = G4RandGauss::shoot(Px,stddevX);
					PyNew = G4RandGauss::shoot(Py,stddevY);
					PzNew = G4RandGauss::shoot(Pz,stddevZ);
				}

			  if(PxNew<Xmin) PxNew=Xmin;
			  if(PyNew<Ymin) PyNew=Ymin;
			  if(PzNew<Zmin) PzNew=Zmin;
			  if(PxNew>Xmax) PxNew=Xmax;
			  if(PyNew>Ymax) PyNew=Ymax;
			  if(PzNew>Zmax) PzNew=Zmax;
			   m_outputDigi->SetLocalPos(G4ThreeVector(PxNew,PyNew,PzNew)); //TC

			   
			  m_outputDigi->SetGlobalPos(m_outputDigi->GetVolumeID().MoveToAncestorVolumeFrame(m_outputDigi->GetLocalPos())); //TC

	    //Getting world Volume
	    // Do not use from TransportationManager as it is not recommended
	    G4Navigator *navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
	    G4VPhysicalVolume *WorldVolume = navigator->GetWorldVolume();
	    m_Navigator = new G4Navigator();
	    m_Navigator->SetWorldVolume(WorldVolume);
			  G4VPhysicalVolume* PV = m_Navigator->LocateGlobalPointAndSetup(m_outputDigi->GetGlobalPos());
			  m_Touchable = m_Navigator->CreateTouchableHistoryHandle();
			  G4int hdepth = m_Touchable->GetHistoryDepth(); // zero always!

			  if ( hdepth == m_systemDepth  )
			  {
				  UpdateVolumeID();
			  }
			  
			  m_OutputDigiCollection->insert(m_outputDigi);



		  }

	  }
	  }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateSpatialResolution::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
		// Ensure the chosen axis configuration is allowed for PET scanners
		if (!(m_nameAxis == "XZ" || m_nameAxis == "YZ")) {
			G4cout << "***ERROR*** GateSpatialResolution::SetSpatialResolutionParameters: "
					  "Only 'XZ' and 'YZ' are allowed as nameAxis values for 2D spatial resolution distributions.\n";
			abort();
		}
	}
  StoreDigiCollection(m_OutputDigiCollection);

}





void GateSpatialResolution::UpdateVolumeID()
{
 for (G4int i=1;i<m_systemDepth;i++)
		{
		G4int CopyNo = m_Touchable->GetReplicaNumber(m_systemDepth-1-i);
		m_outputDigi->ChangeVolumeIDAndOutputVolumeIDValue(i,CopyNo);
		}
}




void GateSpatialResolution::DescribeMyself(size_t indent )
{
	if(m_fwhm)
	  G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_fwhm  << Gateendl;
	else
		G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_fwhmX <<" "<< m_fwhmY<< " "<<m_fwhmZ<< Gateendl;}
