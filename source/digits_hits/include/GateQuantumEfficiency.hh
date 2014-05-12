/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateQuantumEfficiency_h
#define GateQuantumEfficiency_h 1

#include "globals.hh"
#include <iostream>
#include <fstream>
#include <vector>

#include "GateVPulseProcessor.hh"
#include "GateLevelsFinder.hh"

class GateQuantumEfficiencyMessenger;

/*! \class  GateQuantumEfficiency
    \brief  Pulse-processor for simulating a quantum efficiency.

    - GateQuantumEfficiency - by Martin.Rey@epfl.ch (dec 2002)

    - Pulse-processor for simulating the quantum efficiency on each channel of a PM or an APD.
    There are two options: one, is to give an unique quantum efficiency for all the channels,
    and, the other, is to give some lookout tables which contains data and this class takes these data
    with small variations (minus than 2%) for creating the tables before the simulation.

      \sa GateVPulseProcessor
*/
class GateQuantumEfficiency : public GateVPulseProcessor
{
  public:
    //! This function allows to retrieve the current instance of the GateQuantumEfficiency singleton
    /*!
      	If the GateQuantumEfficiency already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateQuantumEfficiency constructor
    */
    static GateQuantumEfficiency* GetInstance(GatePulseProcessorChain* itsChain,
					      const G4String& itsName);

    //! Public Destructor
    virtual ~GateQuantumEfficiency() ;

  private:
    //!< Private constructor which Constructs a new quantum efficiency module attached to a GateDigitizer:
    //! this function should only be called from GetInstance()
    GateQuantumEfficiency(GatePulseProcessorChain* itsChain,
			  const G4String& itsName);

  public:

    //! Check the validity of the volume name where the quantum efficiency will be applied
    void CheckVolumeName(G4String val);

    //! Allow to use file(s) as lookout table for quantum efficiency
    void UseFile(G4String aFile);

    //! Apply an unique quantum efficiency for all the channels
    void SetUniqueQE(G4double val) { m_uniqueQE = val; };

    //! Create the table for the quantum efficiency inhomogeneity
    void CreateTable();

    //! Return the volume name where the quantum efficiency is applied
    G4String GetVolumeName() { return m_volumeName; };

    //! Return the actual QE coef
    G4double GetActQECoeff() { return m_QECoef; };

    //! Return the QE coef
    G4double GetQECoeff(G4int tableNB, G4int crystalNb) { return m_table[tableNB][crystalNb]; };

    //! Return the minimum QE coef
    G4double GetMinQECoeff();

    //! Return the number of element at level 1
    G4int Getlevel1No() { return m_level1No; };

    //! Return the number of element at level 2
    G4int Getlevel2No() { return m_level2No; };

    //! Return the number of element at level 3
    G4int Getlevel3No() { return m_level3No; };

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the gain module
    virtual void DescribeMyself(size_t indent);


  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    //! Give a random number included between 0 and 1
    G4double MonteCarloEngine();
    //! Give a random integer included between 'a and 'b
    size_t MonteCarloInt(size_t a,size_t b);
    //! Give a random number included between 'a and 'b
    G4double MonteCarloG4double(G4double a,G4double b);

  private:
    //! Static pointer to the GateQuantumEfficiency singleton
    static GateQuantumEfficiency* theGateQuantumEfficiency;

    GateQuantumEfficiencyMessenger* m_messenger;       //!< Messenger

    G4String m_volumeName;  //!< Name of the module
    G4int m_testVolume;     //!< equal to 1 if the volume name is valid, 0 else
    size_t m_count;         //!< equal to 0 before first ProcessOnPulse use, then 1
  GateLevelsFinder* m_levelFinder;
    //!< Number of PhysicalVolume copies of the Inserter corresponding @ volume name 'm_volumeName
    G4int m_nbCrystals;
    //!< Number of PhysicalVolume copies of the Inserter corresponding @ level 'm_depth-1
    G4int m_level3No;
    //!< Number of PhysicalVolume copies of the Inserter corresponding @ level 'm_depth-2
    G4int m_level2No;
    //!< Number of PhysicalVolume copies of the Inserter corresponding @ volume name 'm_volumeName
    G4int m_level1No;
    G4int m_nbTables;
    G4int m_volumeIDNo;     //!< numero of the volumeID
    G4int m_i, m_j, m_k;    //!< numero of the volumeID
    size_t m_depth;         //!< Depth of the selected volume in the Inserter
    G4int m_nbFiles;        //!< Number of file(s) used for creating the lookout table
    std::vector<G4String> m_file;  //!< Vector which contains the name(s) of the file(s) for the lookout table
    G4double m_uniqueQE;    //!< Value of the quantum efficiency if it's unique
    G4double** m_table;     //!< Lookout table for the quantum efficiency of all channels
    G4double m_QECoef;      //!< Actual value of the quantum efficiency
    G4double m_minQECoef;   //!< Minimum quantum efficiency
};


#endif
