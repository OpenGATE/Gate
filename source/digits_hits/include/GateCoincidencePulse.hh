/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidencePulse_h
#define GateCoincidencePulse_h 1

#include "GatePulse.hh"
#include "GateHit.hh"

// define the minimum offset for a delayed coincidence window in sec
//#define  MIN_COINC_OFFSET 500.0E-09 // why??? why can't it be any (non-negative) value?

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
        m_coincID=-1;
        m_startTime = firstPulse->GetTime() + itsOffsetWindow;
        m_endTime = m_startTime + itsCoincidenceWindow;
        if(itsOffsetWindow > 0.0)
          m_delayed = true;
        else
          m_delayed = false;
    }

    GateCoincidencePulse(const GateCoincidencePulse& src);

    virtual ~GateCoincidencePulse(){}

    inline G4double GetStartTime() const
      { return m_startTime; }

    inline G4double GetTime() const
      { return m_endTime; }

   inline G4int GetCoincID() const
      { return m_coincID; }
   inline void SetCoincID(int coincID) 
      { m_coincID=coincID; }

    virtual G4bool IsInCoincidence(const GatePulse* newPulse) const;
    virtual G4bool IsAfterWindow(const GatePulse* newPulse) const;

    inline G4bool IsDelayed() const
      { return m_delayed;}

    //printing methods
    //
    friend std::ostream& operator<<(std::ostream&, const GateCoincidencePulse&);

  private:
    G4double m_startTime;
    G4double m_endTime;
    G4bool m_delayed;
    G4int m_coincID;
};

#endif
