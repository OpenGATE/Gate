/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateXMLDocument_hh
#define GateXMLDocument_hh

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>
#include "globals.hh"

class G4MaterialPropertiesTable;

//! Stores the state of the GateXMLDocument class.
struct GateXMLDocumentState
{
  xmlNodePtr cur;
  G4bool     reset;
};

//! Class that allows the user to navigate through a xml document.
class GateXMLDocument
{
  public:
    GateXMLDocument(const G4String& filename);
    ~GateXMLDocument();

    G4bool Ok() const;
    
    G4String GetName();
    G4String GetProperty(const G4String& property);
    G4bool   HasProperty(const G4String& property);
    G4String GetValue();
    
    void   Reset();
    G4bool Enter();
    void   Leave();
    G4bool Next();
    G4bool Previous();
    void   First();
    
    G4bool Find(const G4String& tag);
    G4bool Find(const G4String& tag, const G4String& name);
    G4bool Find(const G4String& tag, const G4String& property, const G4String& value);
    
    GateXMLDocumentState GetState();
    void SetState(GateXMLDocumentState state);
      
  private:
    G4bool     m_ok;
    xmlDocPtr  m_doc;
    xmlNodePtr m_cur;
    G4bool     m_reset;
};

//! Reads a properties table from the xml-file given by doc
/**
  * Looks at the current depth in the xml-file for the first <propertiestable> tag
  * and reads this table. Returns 0 when no table is found
  * */
G4MaterialPropertiesTable* ReadMaterialPropertiesTable(GateXMLDocument* doc);

#endif

#endif
