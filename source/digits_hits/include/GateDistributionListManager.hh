/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionListManager_h
#define GateDistributionListManager_h 1

#include "globals.hh"

#include "GateListManager.hh"

class GateVDistribution;
class GateDistributionListMessenger;

class GateDistributionListManager : public GateListManager
{
  public:
    static GateDistributionListManager* GetInstance();
    static void Init();

    virtual ~GateDistributionListManager(); //!< Public destructor

  private:
    GateDistributionListManager();   //!< Private constructor: this function should only be called from GetInstance()


  public:
    //! \name Methods to manage the list
    //@{
    virtual void RegisterDistribution(GateVDistribution* newDistribution);   //!< Registers a new Distribution in the list
    //@}

    //! \name Methods to retrieve a Distribution
    //@{

    //! Retrieves a Distribution from its name
    inline GateVDistribution* FindDistribution(const G4String& name)
      { return (GateVDistribution*) FindElement(name); }


    //@}

    //! \name Access methods
    //@{
    GateVDistribution* GetDistribution(size_t i)
      {return (GateVDistribution*) GetElement(i);}      	     //!< Retrieves a Distribution from a store-iterator
    //@}


  protected:
    GateDistributionListMessenger* m_messenger;    //!< Pointer to the store's messenger

  private:
    //! Static pointer to the GateDistributionListManager singleton
    static GateDistributionListManager* theGateDistributionListManager;
};


#endif
