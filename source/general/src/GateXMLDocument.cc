/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! Updated on 2012/07/24  by vesna.cuplov@gmail.com
    A warning message is added in case the simulation doesn't load the Materials.xml or Surfaces.xml file.
    I/O warning: This is only a problem when OPTICAL PHOTONS are transported in your simulation.
*/


/**
 * \file GateXMLDocument.cpp
 * \brief Class GateXMLDocument
 */ 

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateXMLDocument.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "GateVSystem.hh"

#include "GateSystemListManager.hh"

//!@name Constructors and destructors
//@{

//! Default constructor.
/**
  * Opens the xml file whose name is given by filename.
  *
  * Use the Ok() method to check if the opening went ok.
  * */
GateXMLDocument::GateXMLDocument(const G4String& filename) :
  m_ok(false), m_reset(true), m_filename(filename)

//
// SJ COMMENTS## : read the file by using a messenger mechanism
//
{

  m_doc = xmlParseFile(filename.c_str());

  if (m_doc)
  {
    m_cur = xmlDocGetRootElement(m_doc);
    if (m_cur==0) xmlFreeDoc(m_doc);
    else m_ok = true;
  }
  else
  {
    std::cout << "I/O warning: Discard the previous warning if your simulation doesn't transport OPTICAL PHOTONS. \n";
    std::cout << "Otherwise, please copy the "<< filename << " file from the gate-source directory in the directory where you run your main macro.\n";
  }
}

  
//! Destructor.
GateXMLDocument::~GateXMLDocument()
{ 
  if (m_doc) xmlFreeDoc(m_doc);
}
//@}

//! Check if file is opened correctly and is ready for read operations
/**
  * Use this check after calling the constructor to check if there were no errors.
  * */
G4bool GateXMLDocument::Ok() const
{ return m_ok;}

//!@name Getting values
//@{

//! Gets the name of the current node.
/**
  * For example, returns "foo" for the node <foo prop="bar">flierp</foo>
  * */
G4String GateXMLDocument::GetName()
{
  return (char *)m_cur->name;
}

//! Gets the value of the property.
/**
  * For example, returns "bar" for the node <foo prop="bar">flierp</foo> when called as GetProperty("prop").
  * Returns an empty string when the property does not exist.
  * */
G4String GateXMLDocument::GetProperty(const G4String& property)
{
  xmlChar*    key = xmlGetProp(m_cur,(const xmlChar*)property.c_str());
  G4String str = key ? (char *)key : "";
#ifdef WIN32
  xmlFreeFunc((void*)key);
#else
  xmlFree(key);
#endif
  return str;
}

//! Checks if the current node has a certain property.
/**
  * For example, returns true for the node <foo prop="bar">flierp</foo> when called as HasProperty("prop").
  * */
G4bool GateXMLDocument::HasProperty(const G4String& property)
{
  xmlChar* key = xmlGetProp(m_cur,(const xmlChar*)property.c_str());
  G4bool     prop = key ? true : false;
#ifdef WIN32
  xmlFreeFunc((void*)key);
#else
  xmlFree(key);
#endif
  return prop;
}

//! Gets the content of the node.
/**
  * For example, returns "flierp" for the node <foo prop="bar">flierp</foo>.
  * */
G4String GateXMLDocument::GetValue()
{
  xmlChar*    key = xmlNodeListGetString(m_doc, m_cur->xmlChildrenNode, 1);
  G4String str = key ? (char *)key : "";
#ifdef WIN32
  xmlFreeFunc((void*)key);
#else
  xmlFree(key);
#endif
  return str;
}
//@}

//!@name Navigation routines
//@{
//! Returns to the root node.
void GateXMLDocument::Reset()
{
  m_cur = xmlDocGetRootElement(m_doc);
  m_reset = true;
}

//! Goes to the first daughter of the current node.
/**
  * Returns false when the node does not contain a daughter.
  * */
G4bool GateXMLDocument::Enter()
{
  if (m_cur->xmlChildrenNode!=0)
  {
    m_cur   = m_cur->xmlChildrenNode;
    m_reset = true;
    return true;
  }
  else return false;
}

//! Goes to the mother of the current node.
void GateXMLDocument::Leave()
{
  m_cur   = m_cur->parent;
  m_reset = false;
}

//! Goes to the next node.
/**
  * Returns false when there is no more node, true otherwise. The method can therefore be used
  * in a while loop:
  * \code
  * while (xmldoc.Next()) 
  * {  
  *   // do something with the node
  * }
  * \endcode
*/
G4bool GateXMLDocument::Next()
{ 
  if (m_cur->next!=0) 
  { 
    m_cur   = m_cur->next;
    m_reset = false;
    return true;
  }
  else 
  {
    m_reset = false;
    return false;
  }
}

//! Goes to the previous node.
/**
  * Returns false when there is no more node, true otherwise. The method can therefore be used
  * in a while loop:
  * \code
  * while (xmldoc.Previous()) 
  * { 
  *   // do something with the node 
  * }
  * \endcode
  * */
G4bool GateXMLDocument::Previous()
{
  if (m_cur->prev!=0)
  {
    m_cur = m_cur->prev;
    return true;
  }
  else
  {
    m_reset = true;
    return false;
  }
}

//! Goes to the first node.
void GateXMLDocument::First()
{
  while (Previous()) ;
  m_reset = true;
}
//@}

//!@name Finding nodes
//@{ 

//! Finds the next node the name given by 'tag'.
/**
  * Returns true when found false otherwise.
  * Find only looks at the current depth.
  * */
G4bool GateXMLDocument::Find(const G4String& tag)
{ 
  if (!m_reset) 
  { if (!Next()) return false;}
  
  do
  { 
    if (GetName()==tag) 
    {
      m_reset = false;
      return true;
    }
  }
  while (Next());
  
  return false;
}

//! Finds the next node the name given by 'tag' and the property name equal to 'value'.
/**
  * Returns true when found false otherwise.
  * Find only looks at the current depth.
  * */
G4bool GateXMLDocument::Find(const G4String& tag, const G4String& name)
{
  return Find(tag,"name",name);
}

//! Finds the next node the name given by 'tag' and the property 'property' equal to 'value'.
/**
  * Returns true when found false otherwise.
  * Find only looks at the current depth.
  * */
G4bool GateXMLDocument::Find(const G4String& tag, const G4String& property, const G4String& value)
{ 
  if (!m_reset) 
  { if (!Next()) return false;}
  
  do
  { 
    if (GetName()==tag)
    { 
      if (GetProperty(property)==value) 
      {
	m_reset = false;
	return true;
      }
    }
  }
  while (Next());
  
  return false;
}
//@}

//! Gets the current position in the document.
/**
  * Can be used in combination with SetState() to return to the previous position in the document.
  * */
GateXMLDocumentState GateXMLDocument::GetState()
{
  GateXMLDocumentState state;
  state.cur   = m_cur;
  state.reset = m_reset;
  return state;
}

//! Sets the position in the document back to the one given by state.
void GateXMLDocument::SetState(GateXMLDocumentState state)
{
  m_cur   = state.cur;
  m_reset = state.reset;
}

// geant4 v11 replaces some properties name by new one
// We propose suggestion for replacements in error message.
const G4String suggest_values_for_keys(const G4String& key)
{
    if(key == "FASTTIMECONSTANT")
        return "SCINTILLATIONTIMECONSTANT1";
    else if( key == "YIELDRATIO")
        return "SCINTILLATIONYIELD1";
    else if( key == "FASTCOMPONENT")
        return "SCINTILLATIONCOMPONENT1";
    return "?";

}



const G4String &GateXMLDocument::GetFilename() const
{
    return m_filename;
}

G4MaterialPropertiesTable* ReadMaterialPropertiesTable(GateXMLDocument* doc)
{
  G4MaterialPropertiesTable* table = 0;
  // read properties table
  doc->First();
  if (doc->Find("propertiestable"))
  {
    doc->Enter();
    table = new G4MaterialPropertiesTable;
    // read through the file and look for properties
    while (doc->Next())
    {
      if (doc->GetName() == "property")
      {
	G4String property = doc->GetProperty("name");
	G4String valuestr = doc->GetProperty("value");
	G4double value    = G4UIcmdWithADouble::GetNewDoubleValue(valuestr.c_str());
	if (doc->HasProperty("unit"))
	{
	  G4String unitstr = "1 " + doc->GetProperty("unit");
	  value *= G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(unitstr.c_str());
	}

    auto list_of_known_properties_key = &table->GetMaterialConstPropertyNames();
    if(std::count(list_of_known_properties_key->begin(), list_of_known_properties_key->end(), property))
    {
        table->AddConstProperty(property.c_str(), value);
    }
    else
    {
        G4cout << "Unknown property '" << property << "'  in xml file '" <<   doc->GetFilename()  << "'. Abort simulation." << G4endl;
        G4cout << "Suggestion: property '"<<  property << "' can be replaced by '" << suggest_values_for_keys(property) << "'" << G4endl;
        exit(-1);

    }


      }
      else if (doc->GetName() == "propertyvector")
      {
	G4String property = doc->GetProperty("name");
	// get unit
	G4double unit = 1;
	if (doc->HasProperty("unit"))
	{
	  G4String unitstr = "1 " + doc->GetProperty("unit");
	  unit = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(unitstr.c_str());
	}
	// get energy unit
	G4double energyunit = 1;
	if (doc->HasProperty("energyunit"))
	{
	  G4String unitstr = "1 " + doc->GetProperty("energyunit");
	  energyunit = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(unitstr.c_str());
	}
	// read vector
	doc->Enter();
	G4MaterialPropertyVector* vector = new G4MaterialPropertyVector;
	while (doc->Find("ve"))
	{
	  G4String energystr = doc->GetProperty("energy");
	  G4double energy    = G4UIcmdWithADouble::GetNewDoubleValue(energystr.c_str());
	  G4String valuestr  = doc->GetProperty("value");
	  G4double value     = G4UIcmdWithADouble::GetNewDoubleValue(valuestr.c_str());
	  vector->InsertValues(energy*energyunit, value*unit);
	}


  auto list_of_known_properties_key = &table->GetMaterialPropertyNames();
  if(std::count(list_of_known_properties_key->begin(), list_of_known_properties_key->end(), property))
  {
      table->AddProperty(property.c_str(), vector);
  }
  else
  {
      G4cout << "Unknown propertyvector '" << property << "'  in xml file '" <<   doc->GetFilename()  << "'. Abort simulation." << G4endl;
      G4cout << "Suggestion: propertyvector '"<<  property << "' can be replaced by '" << suggest_values_for_keys(property) << "'" << G4endl;
      exit(-1);
  }



	doc->Leave();
      }
    }
    doc->Leave();
  }
  return table;
}

#endif
