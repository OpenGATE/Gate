/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "Randomize.hh"

#include "GateCoincidenceSorter.hh"

#include "G4UnitsTable.hh"

#include "GateVolumeID.hh"
#include "GateObjectStore.hh"


#include "GateCoincidenceSorterMessenger.hh"

#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateVSystem.hh"
#include "GateCoincidenceDigiMaker.hh"

//#include <map>

//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::gm_coincSectNum=0;
// Constructs a new coincidence sorter, attached to a GateDigitizer and to a system
GateCoincidenceSorter::GateCoincidenceSorter(GateDigitizer* itsDigitizer,
      	      	      	      	      	     const G4String& itsOutputName,
      	      	      	      	      	     G4double itsWindow,
      	      	      	      	      	     const G4String& itsInputName)
  : GateClockDependent(itsDigitizer->GetObjectName()+"/"+itsOutputName),
    m_digitizer(itsDigitizer),
    m_system(0),
    m_outputName(itsOutputName),
    m_inputName(itsInputName),
    m_coincidenceWindow(itsWindow),
    m_coincidenceWindowJitter(0.),
    m_offset(0.),
    m_offsetJitter(0.),
    m_minSectorDifference(2),
    m_multiplesPolicy(kKeepIfAllAreGoods),
    m_allPulseOpenCoincGate(false),
    m_depth(1),
    m_presortBufferSize(256),
    m_presortWarning(false)
{

  // Create the messenger
  m_messenger = new GateCoincidenceSorterMessenger(this);

  itsDigitizer->InsertDigiMakerModule( new GateCoincidenceDigiMaker(itsDigitizer, itsOutputName,true) );
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Destructor
GateCoincidenceSorter::~GateCoincidenceSorter()
{
  while(m_presortBuffer.size() > 0)
  {
    delete m_presortBuffer.back();
    m_presortBuffer.pop_back();
  }

  while(m_coincidencePulses.size() > 0)
  {
    delete m_coincidencePulses.back();
    m_coincidencePulses.pop_back();
  }

  delete m_messenger;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Overload of the virtual method declared by the base class GateVCoincidenceSorter
// print-out the attributes specific of the sorter
void GateCoincidenceSorter::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Coincidence window:  " << G4BestUnit(m_coincidenceWindow,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence window jitter: " << G4BestUnit(m_coincidenceWindowJitter,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset:  " << G4BestUnit(m_offset,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset jitter: " << G4BestUnit(m_offsetJitter,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Min sector diff.:    " << m_minSectorDifference << Gateendl;
  G4cout << GateTools::Indent(indent) << "Presort buffer size: " << m_presortBufferSize << Gateendl;
  G4cout << GateTools::Indent(indent) << "Input:              '" << m_inputName << "'" << Gateendl;
  G4cout << GateTools::Indent(indent) << "Output:             '" << m_outputName << "'" << Gateendl;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidenceSorter::SetMultiplesPolicy(const G4String& policy)
{
    if (policy=="takeWinnerOfGoods")
    	m_multiplesPolicy=kTakeWinnerOfGoods;
    else if (policy=="takeWinnerIfIsGood")
    	m_multiplesPolicy=kTakeWinnerIfIsGood;
    else if (policy=="takeWinnerIfAllAreGoods")
    	m_multiplesPolicy=kTakeWinnerIfAllAreGoods;
    else if (policy=="killAll")
    	m_multiplesPolicy=kKillAll;
    else if (policy=="takeAllGoods")
    	m_multiplesPolicy=kTakeAllGoods;
    else if (policy=="killAllIfMultipleGoods")
    	m_multiplesPolicy=kKillAllIfMultipleGoods;
    else if (policy=="keepIfAnyIsGood")
    	m_multiplesPolicy=kKeepIfAnyIsGood;
    else if (policy=="keepIfOnlyOneGood")
    	m_multiplesPolicy=kKeepIfOnlyOneGood;
    else if (policy=="keepAll")
    	m_multiplesPolicy=kKeepAll;
    else {
    	if (policy!="keepIfAllAreGoods")
    	    G4cout<<"WARNING : policy not recognized, using default : keepMultiplesIfAllAreGoods\n";
  	m_multiplesPolicy=kKeepIfAllAreGoods;
    }
}
//------------------------------------------------------------------------------------------------------


void GateCoincidenceSorter::ProcessSinglePulseList(GatePulseList* inp)
{
  GatePulse* pulse;
  std::list<GatePulse*>::iterator buf_iter;                // presort buffer iterator
  std::deque<GateCoincidencePulse*>::iterator coince_iter; // coincidence list iterator

  G4bool inCoincidence;

  GateCoincidencePulse* coincidence;
  G4double window, offset;

  GatePulseIterator gpl_iter;      // input pulse list iterator

  if (!IsEnabled())
    return;

  GatePulseList* inputPulseList = inp ? inp : m_digitizer->FindPulseList( m_inputName );

  if (!inputPulseList)
    return ;

  // put input pulses in sorted input buffer
  for(gpl_iter = inputPulseList->begin();gpl_iter != inputPulseList->end();gpl_iter++)
  {
    // make a copy of the pulse
    pulse = new GatePulse(**gpl_iter);

    if(m_presortBuffer.empty())
      m_presortBuffer.push_back(pulse);
    else if(pulse->GetTime() < m_presortBuffer.back()->GetTime())    // check that even isn't earlier than the earliest event in the buffer
    {
      if(!m_presortWarning)
        GateWarning("Event is earlier than earliest event in coincidence presort buffer. Consider using a larger buffer.");
      m_presortWarning = true;
      m_presortBuffer.push_back(pulse); // this will probably not cause a problem, but coincidences may be missed
    }
    else // put the event into the presort buffer in the right place
    {
      buf_iter = m_presortBuffer.begin();
      while(pulse->GetTime() < (*buf_iter)->GetTime())
        buf_iter++;
      m_presortBuffer.insert(buf_iter, pulse);
    }

  }

  //  once buffer reaches the specified size look for coincidences
  for(G4int i = m_presortBuffer.size();i > m_presortBufferSize;i--)
  {
    pulse = m_presortBuffer.back();
    m_presortBuffer.pop_back();

    // process completed coincidence pulse window at front of list
    while(!m_coincidencePulses.empty() && m_coincidencePulses.front()->IsAfterWindow(pulse))
    {
      coincidence = m_coincidencePulses.front();
      m_coincidencePulses.pop_front();

      ProcessCompletedCoincidenceWindow(coincidence);
    }

    // add event to coincidences
    inCoincidence = false;
    coince_iter = m_coincidencePulses.begin();
    while( coince_iter != m_coincidencePulses.end() && (*coince_iter)->IsInCoincidence(pulse) )
    {
      inCoincidence = true;
      (*coince_iter)->push_back(new GatePulse(pulse)); // add a copy so we can delete safely
      coince_iter++;
    }

    // if not after or in the windows, it must be before the rest of coincidence windows
    // so there's no need to check the rest of the coincidence list

    // update coincidence pulse list
    if(m_allPulseOpenCoincGate || !inCoincidence)
    {
      if(m_coincidenceWindowJitter > 0.0)
        window = G4RandGauss::shoot(m_coincidenceWindow,m_coincidenceWindowJitter);
      else
        window = m_coincidenceWindow;

      if(m_offsetJitter > 0.0)
        offset = G4RandGauss::shoot(m_offset,m_offsetJitter);
      else
        offset = m_offset;

      coincidence = new GateCoincidencePulse(m_outputName,pulse,window,offset);
      m_coincidencePulses.push_back(coincidence);
    }
    else
      delete pulse; // pulses that don't open a coincidence window can be discarded
  }

}

// look for valid coincidences
void GateCoincidenceSorter::ProcessCompletedCoincidenceWindow(GateCoincidencePulse *coincidence)
{
  G4int i, j, nPulses;
  G4int nGoods, maxGoods;
  G4double E, maxE;
  G4int winner_i=0;
  G4int winner_j=1;

  G4bool PairWithFirstPulseOnly;

  nPulses = coincidence->size();

  if (nPulses<2)
  {
    delete coincidence;
    return;
  }
  else if (nPulses==2)
  {
    // check if good
    if(IsForbiddenCoincidence(coincidence->at(0),coincidence->at(1)) )
      delete coincidence;
    else
      m_digitizer->StoreCoincidencePulse(coincidence);
    return;
  }
  else // nPulses>2 multiples
  {
    if(m_multiplesPolicy==kKillAll)
    {
      delete coincidence;
      return;
    }

    // if dealing with a delayed window or if other pulses open coincidence windows,
    // we only want to pair with the first pulse to avoid invalid pairs, or double counting
    PairWithFirstPulseOnly = m_allPulseOpenCoincGate | coincidence->IsDelayed();

    if(m_multiplesPolicy==kTakeAllGoods)
    {
      for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++) // iterate over all pairs (single window) or just pairs with initial event (multi-window)
        for(j=i+1; j<nPulses; j++)
          if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)) )
            m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, i, j));
      delete coincidence; // valid pulses extracted so we can delete
      return;
    }

    // count the goods (iterate over all pairs because we're considering the multi as a unit, not breaking it up into pairs)
    nGoods = 0;
    for(i=0; i<(coincidence->IsDelayed()?1:(nPulses-1)); i++)
      for(j=i+1; j<nPulses; j++)
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoods++;

    if( nGoods == 0 )  // all of the remaining options expect at least one good
    {
      delete coincidence;
      return;
    }

    // all the Keep* policies pass on a multi-coincidence rather than breaking into pairs
    if( ( (m_multiplesPolicy==kKeepIfAnyIsGood) /*&& (nGoods>0)*/          ) || // if nGoods = 0, we don't get here
        ( (m_multiplesPolicy==kKeepIfOnlyOneGood) && (nGoods==1)           ) ||
        ( (m_multiplesPolicy==kKeepIfAllAreGoods) && (nGoods==(nPulses*(nPulses-1)/2)) ) )
    {
      m_digitizer->StoreCoincidencePulse(coincidence);
      return; // don't delete the coincidence
    }
    if((m_multiplesPolicy==kKeepIfAnyIsGood)   ||
       (m_multiplesPolicy==kKeepIfOnlyOneGood) ||
       (m_multiplesPolicy==kKeepIfAllAreGoods) )
    {
      delete coincidence;
      return;
    }

    // find winner and count the goods
    maxE = 0.0;
    nGoods = 0;
    for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++)
      for(j=i+1; j<nPulses; j++)
      {
        // this time we might only be counting goods on the subset involving the first event
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoods++;

        E = coincidence->at(i)->GetEnergy() + coincidence->at(j)->GetEnergy();
        if(E>maxE)
        {
          maxE = E;
          winner_i = i;
          winner_j = j;
        }
      }

    if(nGoods==0) // check again, we may have reduced the subset.
    {
      delete coincidence;
      return;
    }

    if(m_multiplesPolicy==kTakeWinnerIfIsGood)
    {
      if(!IsForbiddenCoincidence(coincidence->at(winner_i),coincidence->at(winner_j)) )
        m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
      delete coincidence;
      return;
    }

    if(m_multiplesPolicy==kKillAllIfMultipleGoods)
    {
      if(nGoods>1)
      {
        delete coincidence;
        return;
      } // else find and return the one good event
      else // nGoods==1
      {
        for(i=0; i<(coincidence->IsDelayed()?1:(nPulses-1)); i++)
          for(j=i+1; j<nPulses; j++)
            if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
              m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, i, j));
        delete coincidence;
        return;
      }
    }

    maxGoods = PairWithFirstPulseOnly?(nPulses-1):(nPulses*(nPulses-1)/2);
    if(m_multiplesPolicy==kTakeWinnerIfAllAreGoods)
    {
      if(nGoods==maxGoods)
      {
        m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
        delete coincidence;
        return;
      }
      else
      {
        delete coincidence;
        return;
      }
    }

    if(m_multiplesPolicy==kTakeWinnerOfGoods)
    {
      // find winner
      maxE = 0.0;
      for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++)
        for(j=i+1; j<nPulses; j++)
        {
          if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          {
            E = coincidence->at(i)->GetEnergy() + coincidence->at(j)->GetEnergy();
            if(E>maxE)
            {
              maxE = E;
              winner_i = i;
              winner_j = j;
            }
          }
        }
      m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
      delete coincidence; // valid pulses extracted so we can delete
      return;
    }
  }

  delete coincidence;
  return;

}

GateCoincidencePulse* GateCoincidenceSorter::CreateSubPulse(GateCoincidencePulse* coincidence, G4int i, G4int j)
{
  GatePulse* pulse1 = new GatePulse(coincidence->at(i));
  GatePulse* pulse2 = new GatePulse(coincidence->at(j));
  G4double offset = coincidence->GetStartTime() - pulse1->GetTime();
  GateCoincidencePulse *newCoincPulse = new GateCoincidencePulse(m_outputName,pulse1,m_coincidenceWindow,offset);
  newCoincPulse->push_back(pulse2);
  return newCoincPulse;
}

//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::ComputeSectorID(const GatePulse& pulse)
{
    if (m_depth>=(G4int)pulse.GetOutputVolumeID().size()) {
    	G4cerr<<"[GateCoincidenceSorter::ComputeSectorID]: Required depth's too deep, setting it to 1\n";
	m_depth=1;
    }
    static std::vector<G4int> gkSectorMultiplier;
    static std::vector<G4int> gkSectorNumber;
    if (gkSectorMultiplier.empty()){
    	// this code is done just one time for performance improving
	// one suppose that the system hierarchy is linear until the desired depth
    	GateSystemComponent* comp = m_system->GetBaseComponent();
	G4int depth=0;
	while (comp){
	    gkSectorNumber.push_back(comp->GetAngularRepeatNumber());
	    if ( (depth<m_depth) && ( comp->GetChildNumber() == 1)   ){
	    	comp = comp->GetChildComponent(0);
		depth++;
	    }
	    else
	    	comp=0;
	}
	gkSectorMultiplier.resize(gkSectorNumber.size());
	gkSectorMultiplier[gkSectorNumber.size()-1] = 1;
	for (G4int i=(G4int)gkSectorNumber.size()-2;i>=0;--i){
	    gkSectorMultiplier[i] = gkSectorMultiplier[i+1] * gkSectorNumber[i+1];
	}
	gm_coincSectNum = gkSectorMultiplier[0];
    }
    G4int ans=0;
    for (G4int i=0;i<=m_depth;i++){
    	G4int x = pulse.GetComponentID(i)%gkSectorNumber[i];
    	ans += x*gkSectorMultiplier[i];
    }

    return ans;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Check whether a coincidence is invalid: ring difference or sector difference too small...
G4bool GateCoincidenceSorter::IsForbiddenCoincidence(const GatePulse* pulse1, const GatePulse* pulse2)
{
  G4int blockID1 = m_system->GetMainComponentID(pulse1),
        blockID2 = m_system->GetMainComponentID(pulse2);

 // Modif by D. Lazaro, February 25th, 2004
  // Computation of sectorID, sectorNumber and sectorDifference, paramaters depending on
  // the geometry construction of the scanner (spherical for system ecatAccel and cylindrical
  // for other systems as Ecat, CPET and cylindricalPET)

  const G4String name = m_system->GetName();
  G4String nameComp = "systems/ecatAccel";
  //G4cout << "NAME OF THE SYSTEM: " << name << "; NAME TO COMPARE: " << nameComp << Gateendl;
  int comp = strcmp(name,nameComp);

  if (comp == 0) {
    // Compute the sector difference
    G4int sectorID1 = m_system->ComputeSectorIDSphere(blockID1),
    sectorID2 = m_system->ComputeSectorIDSphere(blockID2);

    // Get the number of sectors per ring
    G4int sectorNumber = m_system->GetCoincidentSectorNumberSphere();

    // Deal with the circular difference problem
    G4int sectorDiff1 = sectorID1 - sectorID2;
    if (sectorDiff1<0)
      sectorDiff1 += sectorNumber;
    G4int sectorDiff2 = sectorID2 - sectorID1;
    if (sectorDiff2<0)
      sectorDiff2 += sectorNumber;
    G4int sectorDifference = std::min(sectorDiff1,sectorDiff2);

    //Compare the sector difference with the minimum differences for valid coincidences
    if (sectorDifference<m_minSectorDifference) {
      if (nVerboseLevel>1)
        G4cout << "[GateCoincidenceSorter::IsForbiddenCoincidence]: coincidence between neighbor blocks --> refused\n";
      return true;
    }
    return false;
  }
  else {
  // Compute the sector difference
  G4int sectorID1 = ComputeSectorID(*pulse1),
      	sectorID2 = ComputeSectorID(*pulse2);

  // Get the number of sectors per ring
  // G4int sectorNumber = GetCoincidentSectorNumber();
  // Deal with the circular difference problem
  G4int sectorDiff1 = sectorID1 - sectorID2;
  if (sectorDiff1<0)
    sectorDiff1 += gm_coincSectNum;
  G4int sectorDiff2 = sectorID2 - sectorID1;
  if (sectorDiff2<0)
    sectorDiff2 += gm_coincSectNum;
  G4int sectorDifference = std::min(sectorDiff1,sectorDiff2);

  //Compare the sector difference with the minimum differences for valid coincidences
  if (sectorDifference<m_minSectorDifference) {
      	if (nVerboseLevel>1)
      	    G4cout << "[GateCoincidenceSorter::IsForbiddenCoincidence]: coincidence between neighbour blocks --> refused\n";
	return true;
  }

  return false;
  }
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Next method was added for the multi-system approach

void GateCoincidenceSorter::SetSystem(G4String& inputName)
{
   for (size_t i=0; i<m_digitizer->GetChainNumber() ; ++i)
   {
      G4String pPCOutputName = m_digitizer->GetChain(i)->GetOutputName();

      if(pPCOutputName.compare(inputName) == 0)
      {
         this->SetSystem(m_digitizer->GetChain(i)->GetSystem());
         break;
      }
   }

}
