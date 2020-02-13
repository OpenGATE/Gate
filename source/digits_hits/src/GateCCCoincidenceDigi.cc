

#include "GateCCCoincidenceDigi.hh"
#include "G4UnitsTable.hh"
#include "GatePulse.hh"
#include <iomanip>



G4Allocator<GateCCCoincidenceDigi> GateCCCoincidenceDigiAllocator;




GateCCCoincidenceDigi::GateCCCoincidenceDigi()
  : m_pulse()
{
    coincID=0;
}



GateCCCoincidenceDigi::GateCCCoincidenceDigi(GatePulse* pulse, G4int coincidenceID)
  : m_pulse(*pulse), coincID(coincidenceID)
{
}


GateCCCoincidenceDigi::GateCCCoincidenceDigi(const GatePulse& pulse,G4int coincidenceID)
  : m_pulse(pulse), coincID(coincidenceID)
{
}




void GateCCCoincidenceDigi::Draw()
{;}





void GateCCCoincidenceDigi::Print()
{

  G4cout << this << Gateendl;

}




