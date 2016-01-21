/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidencePulse_h
#define GateCoincidencePulse_h 1

#include "GatePulse.hh"
#include "GateCrystalHit.hh"

// define the minimum offset for a delayed coincidence window in sec
#define  MIN_COINC_OFFSET 500.0E-09 // why??? why can't it be any (non-negative) value?

class GateCoincidencePulse : public GatePulseList
{
  public:
    inline GateCoincidencePulse(const G4String& itsName,
                                GatePulse *firstPulse,
                                G4double itsCoincidenceWindow,
                                G4double itsOffsetWindow)
      : GatePulseList(itsName)
    {
        push_back(firstPulse);
        m_startTime = firstPulse->GetTime() + itsOffsetWindow;
        m_endTime = m_startTime + itsCoincidenceWindow;
    }

    GateCoincidencePulse(const GateCoincidencePulse& src);

    virtual ~GateCoincidencePulse(){}

    inline G4double GetStartTime() const
      { return m_startTime; }

    inline G4double GetTime() const
      { return m_endTime; }

    virtual G4bool IsInCoincidence(const GatePulse* newPulse) const;
    virtual G4bool IsAfterWindow(const GatePulse* newPulse) const;

    //printing methods
    //
    friend std::ostream& operator<<(std::ostream&, const GateCoincidencePulse&);

  private:
    G4double m_startTime;
    G4double m_endTime;
};

#endif
