/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for virtual crystals, needed to simulate ecat like sinogram output for Biograph scanners

----------------------*/

#include "GateSinogram.hh"

#include "globals.hh"
#include "G4UnitsTable.hh"

#include "Randomize.hh"

// Reset the matrix and prepare a new acquisition
void GateSinogram::Reset(size_t ringNumber, size_t crystalNumber, size_t radialElemNb, size_t virtualRingNumber, size_t virtualCrystalPerBlockNumber)
{
   size_t sinoID;

  // Fist clean-up the result of a previous acqisition (if any)
  if (m_data) {
    for (sinoID=0;sinoID<m_sinogramNb;sinoID++) {
      free(m_data[sinoID]);
    }
    free(m_data);
    m_data=0;
  }
  if (m_randomsNb) {
    free(m_randomsNb);
    m_randomsNb=0;
  }

  // Store the new number of sinograms
  m_ringNb = ringNumber;
  m_crystalNb = crystalNumber;
  m_radialElemNb = radialElemNb;
  m_sinogramNb = ringNumber*ringNumber;
  m_currentFrameID = -1;
  m_currentGateID = -1;
  m_currentDataID = -1;
  m_currentBedID = -1;

  // C. Comtat, February 2011: Required to simulate Biograph output sinograms with virtual crystals
  m_virtualRingPerBlockNb = virtualRingNumber;
  m_virtualCrystalPerBlockNb = virtualCrystalPerBlockNumber;

  if (!m_ringNb || !m_crystalNb || !m_radialElemNb) {
    return;
  }

  if (nVerboseLevel > 2) {
    G4cout << " >> Allocating " << m_sinogramNb << " 2D sinograms of " << m_radialElemNb <<
              " radial element X " << m_crystalNb/2 << " views each" << G4endl;
  }
  // Allocate the data pointer
  m_data = (SinogramDataType**) malloc( m_sinogramNb * sizeof(SinogramDataType*) );
  if (!m_data) {
    G4Exception( "GateSinogram::Reset", "Reset", FatalException, "Could not allocate a 2D sinogram set (out of memory?)\n");
  }
  // Do the allocations for each 2D sinogram
  for (sinoID=0;sinoID<m_sinogramNb;sinoID++) {
    m_data[sinoID] = (SinogramDataType*) malloc( BytesPerSinogram() );
    if (!(m_data[sinoID])) {
     G4Exception( "GateSinogram::Reset", "Reset", FatalException, "Could not allocate a new 2D sinogram (out of memory?)\n");
    }
  }
  // Allocate the randoms pointer
  m_randomsNb = (SinogramDataType*) calloc( m_sinogramNb , sizeof(SinogramDataType) );
  if (!m_randomsNb) G4Exception( "GateSinogram::Reset", "Reset", FatalException, "Could not allocate a new randoms array (out of memory?)\n");
}


// Clear the matrix and prepare a new run
void GateSinogram::ClearData(size_t frameID, size_t gateID, size_t dataID, size_t bedID)
{
  size_t sinoID;

  // Store the 4D sinogram ID
  m_currentFrameID = frameID;
  m_currentGateID = gateID;
  m_currentDataID = dataID;
  m_currentBedID = bedID;

  // Clear the data sets
  if (nVerboseLevel > 2) {
    G4cout << " >> Reseting " << m_sinogramNb << " 2D sinograms to 0 " << G4endl;
    G4cout << "    for frame " << m_currentFrameID << ", gate " << m_currentGateID <<
              ", data " << m_currentDataID << ", bed " << m_currentBedID << G4endl;
  }
  for (sinoID=0;sinoID<m_sinogramNb;sinoID++)
    memset(m_data[sinoID],0, BytesPerSinogram() );
  memset(m_randomsNb,0,m_sinogramNb * sizeof(SinogramDataType));
}

G4int GateSinogram::GetSinoID( G4int ring1ID, G4int ring2ID)
{
  G4int  DeltaZ,ADeltaZ,sinoID,i;
  // Check that the IDs are valid
  if ( (ring1ID<0) || (ring1ID>=(G4int) m_ringNb) ) {
    G4cerr << "[GateToSinogram::GetSinoID]:" << G4endl
      	   << "Received a wrong ring-1 ID (" << ring1ID << "): ignored!" << G4endl;
    return -1;
  }
  if ( (ring2ID<0) || (ring2ID>=(G4int) m_ringNb) ) {
    G4cerr << "[GateToSinogram::GetSinoID]:" << G4endl
      	   << "Received a wrong ring-2 ID (" << ring2ID << "): ignored!" << G4endl;
    return -2;
  }
  // original: sinoID = ring1ID + ring2ID*m_ringNb;
  DeltaZ = ring2ID-ring1ID;
  if (DeltaZ < 0) ADeltaZ = -DeltaZ; else ADeltaZ = DeltaZ;
  sinoID = (ring1ID+ring2ID-ADeltaZ)/2;
  if (ADeltaZ > 0) sinoID += m_ringNb;
  if (ADeltaZ > 1) for (i=1;i<ADeltaZ;i++) sinoID += 2*(m_ringNb-i);
  if (DeltaZ < 0) sinoID += m_ringNb-ADeltaZ;
  return sinoID;
}

G4int GateSinogram::FillRandoms( G4int ring1ID, G4int ring2ID)
{
  G4int sinoID;
  sinoID = GetSinoID(ring1ID,ring2ID);
  // Check that the ID is valid
  if ( (sinoID<0) || (sinoID>=(G4int) m_sinogramNb) ) {
    G4cerr << "[GateToSinogram::FillRandoms]:" << G4endl
      	   << "Received a hit with wrong ring IDs (" << ring1ID << ","<< ring2ID << "): ignored!" << G4endl;
    return -2;
  }
  SinogramDataType& dest = m_randomsNb[sinoID];
  if (dest<SHRT_MAX) {
    dest++;
  } else {
    G4cerr  << "[GateSinogram]: bin of 2D sinogram " << sinoID << " for randoms has reached its maximum value (" << SHRT_MAX
            << "): hit will be lost!" << G4endl;
    return -7;
  }
  return 0;
}

void GateSinogram::CrystalBlurring( G4int *ringID, G4int *crystalID, G4double ringResolution, G4double crystalResolution)
{
  if (ringResolution > 0.) {
    G4double ringNewID    = G4RandGauss::shoot((G4double) ringID[0],ringResolution/2.35);
    ringID[0] = (G4int) (ringNewID + 0.5);
    if (ringID[0] < 0) ringID[0] = 0;
    else if (ringID[0] >= (G4int) m_ringNb) ringID[0] = m_ringNb - 1;
  }
  if (crystalResolution > 0.) {
    G4double crystalNewID = G4RandGauss::shoot((G4double) crystalID[0],crystalResolution/2.35);
    crystalID[0] = (G4int) (crystalNewID +  0.5);
    if (crystalID[0] < 0) crystalID[0] = m_crystalNb + crystalID[0];
    else if (crystalID[0] >= (G4int) m_crystalNb) crystalID[0] = crystalID[0] - m_crystalNb;
  }
}

// Store a digi into a projection
G4int GateSinogram::Fill( G4int ring1ID, G4int ring2ID, G4int crystal1ID, G4int crystal2ID, int signe)
{

  size_t  binElemID, binViewID;
  G4int   sinoID,det1_c,diff1,diff2,sigma,itemp;
  //G4int det2_c;
	sinoID = GetSinoID(ring1ID,ring2ID);
  if (nVerboseLevel > 3) {
    G4cout << " >> [GateSinogram::Fill]: rings " << ring1ID << "," << ring2ID  << " give sino ID " << sinoID << G4endl;
  }
  // Check that the IDs are valid
  if ( (sinoID<0) || (sinoID>=(G4int) m_sinogramNb) ) {
    G4cerr << "[GateSinogram::Fill]:" << G4endl
      	   << "Received a hit with wrong ring IDs (" << ring1ID << ","<< ring2ID << "): ignored!" << G4endl;
    return -2;
  }
  if ( (crystal1ID<0) || (crystal1ID>=(G4int) m_crystalNb) ) {
    G4cerr << "[GateToSinogram::Fill]:" << G4endl
      	   << "Received a hit with a wrong crystal one ID (" << crystal1ID << "): ignored!" << G4endl;
    return -3;
  }
  if ( (crystal2ID<0) || (crystal2ID>=(G4int) m_crystalNb) ) {
    G4cerr << "[GateToSinogram::Fill]:" << G4endl
      	   << "Received a hit with a wrong crystal two ID (" << crystal2ID << "): ignored!" << G4endl;
    return -4;
  }


  itemp = ((crystal1ID + crystal2ID + (m_crystalNb/2)+1)/2) % (m_crystalNb/2);
  if  ( (itemp<0) || (itemp>=(G4int)m_crystalNb/2) ) {
    if (nVerboseLevel > 3)
      G4cerr << "[GateSinogram]: view ID (" << itemp << ") outside the sinogram boundaries ("
	     << "0" << "-" << m_crystalNb/2-1 << "); event ignored!" << G4endl;
    return -5;
  }
  binViewID = itemp;

  det1_c = binViewID;
  //det2_c = binViewID + (m_crystalNb/2);
  if (fabs(crystal1ID - det1_c) < fabs(crystal1ID - (det1_c + (G4int)m_crystalNb)))
    diff1 = crystal1ID - det1_c;
  else
    diff1 = crystal1ID - (det1_c + m_crystalNb);
  if (fabs(crystal2ID - det1_c) < fabs(crystal2ID - (det1_c + (G4int)m_crystalNb)))
    diff2 = crystal2ID - det1_c;
  else
    diff2 = crystal2ID - (det1_c + m_crystalNb);
  if (fabs(diff1) < fabs(diff2)) sigma = crystal1ID - crystal2ID;
  else sigma = crystal2ID - crystal1ID;
  if (sigma < 0)  sigma += m_crystalNb;
  // m_elemNb :=  m_crystalNb/2
  // m_viewNb :=  m_crystalNb/2
  itemp = sigma + (m_radialElemNb)/2 - m_crystalNb/2;
  if  ( (itemp<0) || (itemp>=(G4int)m_radialElemNb) ) {
    if (nVerboseLevel > 3) {
      G4cerr << "[GateSinogram]: radial element ID (" << itemp << ") outside the sinogram boundaries ("
	     << "0" << "-" << m_radialElemNb-1 << "); event ignored!" << G4endl;
      G4cerr << "                 crystal1 ID = " << crystal1ID << " ; crystal2 ID = " << crystal2ID << G4endl;
      G4cerr << "                 bin view ID = " << binViewID << G4endl;
    }
    return -6;
  }
  binElemID = itemp;

  // Increment the appropriate bin (provided that we've not reached the top)
  if (nVerboseLevel > 3)
      G4cout << " >> [GateSinogram::Fill]: binning LOR at (" <<  crystal1ID << "," << ring1ID << ")-(" << crystal2ID  << ","
      << ring2ID << ") into sinogram bin (" << binElemID << "," << binViewID <<
      ") of 2D sinogram (" << ring1ID+ring2ID << "," << ring2ID-ring1ID << ")" << G4endl;
  SinogramDataType& dest = m_data[sinoID][ binElemID + binViewID * m_radialElemNb];

  if (signe > 0) {
    dest++;
  } else if (signe < 0) {
    dest--;
  }
  else /*if (signe == 0)*/ {
    G4cerr <<   "[GateSinogram::Fill]: filling signe not provided" << G4endl;
    return -8;
  }
  /*if (dest>=SHRT_MAX || dest<=SHRT_MIN) {
    G4cerr  << "[GateSinogram]: bin (" << binElemID << "," << binViewID << ") of 2D sinogram " << sinoID << " has reached its maximum value (" << SHRT_MAX << "): hit will be lost!" << G4endl;
    return -7;
  }*/
  return 0;
}



/* Writes a 2D sinogram onto an output stream

   dest:    	  the destination stream
   sinoID:    	  the 2D sinogram whose data to stream-out
*/
void GateSinogram::StreamOut(std::ofstream& dest, size_t sinoID, size_t seekID)
{
    if (sinoID >= m_sinogramNb) G4Exception( "GateSinogram::StreamOut", "StreamOut", FatalException, "SinoID out of range !\n");
    dest.seekp(seekID * BytesPerSinogram(),std::ios::beg);
    if ( dest.bad() ) G4Exception( "GateSinogram::StreamOut", "StreamOut", FatalException, "Could not write a 2D sinogram onto the disk (out of disk space?)!\n");
    dest.write((const char*)(m_data[sinoID]),BytesPerSinogram() );
    if ( dest.bad() ) G4Exception( "GateToSinogram:StreamOut", "StreamOut", FatalException, "Could not write a 2D sinogram onto the disk (out of disk space?)!\n");
    dest.flush();
}
