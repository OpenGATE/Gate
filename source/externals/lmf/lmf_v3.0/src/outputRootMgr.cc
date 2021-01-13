/*-------------------------------------------------------

List Mode Format 
                        
--  outputRootMgr.c  --                      

Martin.Rey@epfl.ch
Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of outputRootMgr.c:
This module allows to write in root file some LMF infos
of Event record only 

-------------------------------------------------------*/

#include <iostream>
#include <stdio.h>
#include "lmf.h"

#include "TFile.h"
#include "TTree.h"



static int doneOnce = FALSE;
static TTree *lmfTree;
static TFile *pRootFile = NULL;



void outputRoot(const ENCODING_HEADER * pEncoH,
		const EVENT_HEADER * pEH,
		const GATE_DIGI_HEADER * pGDH, const EVENT_RECORD * pER)
{
  Double_t ownTime;
  static Int_t nevt = 0;
  Double_t timeMillis = 0;
  Float_t timeOfFlight = 0;

  u16 *pcrist;
  Int_t gantryAngularPos, gantryAxialPos;
  Int_t sector, module, submodule, crystal, layer;
  Int_t sector1, module1, submodule1, crystal1, layer1;

  calculOfEventPosition resultOfCalculOfEventPosition;	/* JMV modif */
  Float_t radial, tangential, axial;

  Int_t energy, energy1;
  Int_t neighb;

  static int fileOK = TRUE;	/* // the file is a coincidence file ? */

  GATE_DIGI_RECORD *pGDR;

  if (doneOnce == FALSE) {
    pRootFile = new TFile("lmf.root", "RECREATE");
    if (pEH->coincidenceBool) {
      lmfTree = new TTree("coinci", "Coincidences Tree");
      lmfTree->Branch("nevt", &nevt, "nevt/I");
      lmfTree->Branch("timeMillis", &timeMillis, "timeMillis/D");
      lmfTree->Branch("timeOfFlight", &timeOfFlight, "timeOfFlight/F");
      lmfTree->Branch("rsector", &sector, "rsector/I");
      lmfTree->Branch("module", &module, "module/I");
      lmfTree->Branch("submodule", &submodule, "submodule/I");
      lmfTree->Branch("crystal", &crystal, "crystal/I");
      lmfTree->Branch("layer", &layer, "layer/I");
      lmfTree->Branch("rsector1", &sector1, "rsector1/I");
      lmfTree->Branch("module1", &module1, "module1/I");
      lmfTree->Branch("submodule1", &submodule1, "submodule1/I");
      lmfTree->Branch("crystal1", &crystal1, "crystal1/I");
      lmfTree->Branch("layer1", &layer1, "layer1/I");
      if (pEH->energyBool == TRUE) {
	lmfTree->Branch("energy", &energy, "energy/I");
	lmfTree->Branch("energy1", &energy1, "energy1/I");
      }
      if (pEH->gantryAngularPosBool == 1)
	lmfTree->Branch("gantryAngularPos", &gantryAngularPos,
			"gantryAngularPos/I");
      if (pEH->gantryAxialPosBool == 1)
	lmfTree->Branch("gantryAxialPos", &gantryAxialPos,
			"gantryAxialPos/I");
    } else {
      lmfTree = new TTree("singles", "Singles Tree");
      lmfTree->Branch("nevt", &nevt, "nevt/I");
      lmfTree->Branch("time", &ownTime, "time/D");
      lmfTree->Branch("sector", &sector, "sector/I");
      lmfTree->Branch("module", &module, "module/I");
      lmfTree->Branch("submodule", &submodule, "submodule/I");
      lmfTree->Branch("crystal", &crystal, "crystal/I");
      lmfTree->Branch("layer", &layer, "layer/I");
      lmfTree->Branch("radial", &radial, "radial/F");
      lmfTree->Branch("tangential", &tangential, "tangential/F");
      lmfTree->Branch("axial", &axial, "axial/F");
      if (pEH->energyBool == TRUE)
	lmfTree->Branch("energy", &energy, "energy/I");
      lmfTree->Branch("neighb", &neighb, "neighb/I");
      if (pEH->gantryAngularPosBool == 1)
	lmfTree->Branch("gantryAngularPos", &gantryAngularPos,
			"gantryAngularPos/I");
      if (pEH->gantryAxialPosBool == 1)
	lmfTree->Branch("gantryAxialPos", &gantryAxialPos,
			"gantryAxialPos/I");
    }

    if (pEncoH == NULL) {
      printf
	("*** warning : outputRootMgr.c : no encoding pointer pEncoH defined, please check \n");
      exit(EXIT_FAILURE);
    }
    else {
      if (pEncoH->scanContent.eventRecordBool == FALSE) {
	printf("*** error : outputRootMgr.c : not an event record file\n");
	exit(EXIT_FAILURE);
      }
      if (pEncoH->scanContent.gateDigiRecordBool == FALSE)
	printf
	    ("*** warning : outputRootMgr.c : no gate digi in this file\n");
    }
    if (pGDH == NULL)
      printf
	  ("*** warning : outputRootMgr.c : no encoding pointer pGDH defined, please check \n");
    else {
      if (pGDH->comptonBool == FALSE)
	printf
	    ("*** warning : outputRootMgr.c : no number of compton in this file\n");
      if (pGDH->eventIDBool == FALSE)
	printf
	    ("*** warning : outputRootMgr.c : no event ID  in this file\n");
    }
    doneOnce = TRUE;
  }

  if (fileOK) {
    // coincidences --------------------------------------------------------
    if (pEH->coincidenceBool) {
      nevt++;
      timeMillis = (Double_t) getTimeOfThisCOINCI(pER);
      timeOfFlight = getTimeOfFlightOfThisCOINCI(pER);

      // Crystal address
      pcrist = demakeid(pER->crystalIDs[0], pEncoH);
      sector = (Int_t) pcrist[4];
      module = (Int_t) pcrist[3];
      submodule = (Int_t) pcrist[2];
      crystal = (Int_t) pcrist[1];
      layer = (Int_t) pcrist[0];

      // Crystal address
      pcrist = demakeid(pER->crystalIDs[1], pEncoH);
      sector1 = (Int_t) pcrist[4];
      module1 = (Int_t) pcrist[3];
      submodule1 = (Int_t) pcrist[2];
      crystal1 = (Int_t) pcrist[1];
      layer1 = (Int_t) pcrist[0];

      if (pEH->energyBool == TRUE) {
	energy = getEnergyStepFromCCH() * (Int_t) (pER->energy[0]);
	energy1 = getEnergyStepFromCCH() * (Int_t) (pER->energy[1]);
      }

      if (pEH->gantryAngularPosBool == 1)
	gantryAngularPos = pER->gantryAngularPos;

      if (pEH->gantryAxialPosBool == 1)
	gantryAxialPos = pER->gantryAxialPos;

      lmfTree->Fill();
    }
    // singles ------------------------------------------------------------
    else {
      nevt++;
      ownTime = (Double_t) (getTimeOfThisEVENT(pER) / 1e3);	// time in nanos (new)
      /*   time_in_picos    crystalID    eventID */
      if (pGDR != NULL) {
      }

      /* time_in_millis / TOF / crystalIDS0  crystalID1 / EventID0 /  EventID1 */
      timeMillis =
	  (Double_t) (pER->timeStamp[2] + 256 * pER->timeStamp[1] +
		      256 * 256 * pER->timeStamp[0]);

      /* MOdif to dump crystals XYZ positions in lab frame JMV/30-4-03 */

      // Crystal address
      pcrist = demakeid(pER->crystalIDs[0], pEncoH);
      sector = (Int_t) pcrist[4];
      module = (Int_t) pcrist[3];
      submodule = (Int_t) pcrist[2];
      crystal = (Int_t) pcrist[1];
      layer = (Int_t) pcrist[0];

      // display XYZ pos for an event record, adapted from processRecordCarrier
      resultOfCalculOfEventPosition =
	  locateEventInLaboratory(pEncoH, pER, 0);
      radial =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.radial;
      tangential =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.
	  tangential;
      axial =
	  resultOfCalculOfEventPosition.eventInLaboratory3DPosition.axial;

      if (pEH->energyBool == TRUE)
	energy = getEnergyStepFromCCH() * (Int_t) (pER->energy[0]);

      neighb = pER->fpgaNeighInfo[0];

      lmfTree->Fill();
    }
    free(pcrist);
  }
}

void destroyOutputRootMgr()
{
  lmfTree->Print();

  pRootFile->Write();
  pRootFile->Close();
  doneOnce = FALSE;
}
