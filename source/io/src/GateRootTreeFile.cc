//
// Created by mdupont on 13/03/18.
//

#include "GateRootTreeFile.hh"
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "TLeaf.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include "GateTreeFileManager.hh"
#include "GateFileExceptions.hh"

using namespace std;


GateRootTree::GateRootTree() : GateTree()
{

//  m_tmapOfDefinition[typeid(Char_t)] = "C";

  add_leaflist<Char_t>("B", "Char_t");
  add_leaflist<UChar_t>("b", "UChar_t");
  add_leaflist<Short_t>("S", "Short_t");
  add_leaflist<UShort_t>("s", "UShort_t");
  add_leaflist<Int_t>("I", "Int_t");
  add_leaflist<UInt_t>("i", "UInt_t");
  add_leaflist<Float_t>("F", "Float_t");
  add_leaflist<Double_t>("D", "Double_t");
  add_leaflist<Long64_t>("L", "Long64_t");
  add_leaflist<ULong64_t>("l", "ULong64_t");
  add_leaflist<Bool_t>("O", "Bool_t");

  add_leaflist<string>("C", "");
  add_leaflist<char*>("C", "");
}

void GateRootTree::register_variable(const std::string &name, const void *p, std::type_index t_index)
{
  void *pp = (void*)p;

  std::stringstream leaf_ss;

  auto b = name.find_first_of("[");
  leaf_ss << name.substr(0, b) << "/" << m_tmapOfDefinition.at(t_index);
  string leaf_s = leaf_ss.str();

//  cout << "RootTreeFile::register_variable name = " << name << " t_index = " << t_index.name() << " leaf_s = " << leaf_s << "\n";


  m_ttree->Branch(name.c_str(), pp, leaf_s.c_str());
//  m_ttree->Branch(name.c_str(), pp);

}

void GateRootTree::register_variable(const std::string&, const std::string*, size_t )
{

}

void GateRootTree::register_variable(const std::string &name, const char *p, size_t)
{
  register_variable(name, p, typeid(string));

}

void GateRootTree::register_variable(const std::string &name, const int *p, size_t n)
{
   void *pp = (void*)p;

   std::stringstream leaf_ss;
   leaf_ss << name <<"["<<n <<"]/" <<"I";
   string leaf_s = leaf_ss.str();
   m_ttree->Branch(name.c_str(), pp, leaf_s.c_str());
}

void GateOutputRootTreeFile::open(const std::string& s)
{
  GateFile::open(s.c_str(), ios_base::out);
  m_file = new TFile(s.c_str(), "RECREATE");
//  cout << "create tree name = " << m_nameOfTree << " from file " << endl;
  m_ttree = new TTree(m_nameOfTree.c_str(), m_nameOfTree.c_str());
}

void GateInputRootTreeFile::open(const std::string& s)
{
  GateFile::open(s.c_str(), ios_base::in);
  m_file = new TFile(s.c_str(), "READ");
  if(m_file->IsZombie())
  {
    std::stringstream ss;
    ss << "Error opening file! '"  << s <<  "' : " << strerror(errno) ;
    throw std::ios::failure(ss.str());
  }



}

void GateOutputRootTreeFile::write()
{
  m_file->Write();
}

void GateRootTree::close()
{
  m_file->Close();
  delete m_file;
}

void GateOutputRootTreeFile::close()
{
  GateOutputRootTreeFile::write();
  GateRootTree::close();
}

void GateInputRootTreeFile::close()
{
  GateRootTree::close();
}


bool GateOutputRootTreeFile::is_open()
{
    return m_file->IsOpen();
}

bool GateInputRootTreeFile::is_open()
{
  return m_file->IsOpen();
}

void GateOutputRootTreeFile::fill()
{

  for(auto ss: m_mapConstStringToRootString)
  {
//    ss.second->assign(*ss.first);
    strcpy(ss.second, ss.first->data());
  }

  m_ttree->Fill();
}

void GateOutputRootTreeFile::write_header()
{

}

void GateOutputRootTreeFile::write_variable(const std::string &name, const void *p, std::type_index t_index)
{
    this->register_variable(name, p, t_index);
}
void GateOutputRootTreeFile::write_variable(const std::string &name, const std::string *p, size_t nb_char)
{
    char *s = new char[nb_char];
    m_mapConstStringToRootString.emplace(p, s);


    this->write_variable(name, s, nb_char);
}
void GateOutputRootTreeFile::write_variable(const std::string &name, const char *p, size_t nb_char)
{
    this->register_variable(name, p, nb_char);
}

void GateOutputRootTreeFile::write_variable(const std::string &name, const int *p, size_t n)
{
    this->register_variable(name, p, n);
}

void GateInputRootTreeFile::check_existence_and_kind(const std::string &name, std::type_index t_index)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  assert(m_ttree);
  auto leaf = m_ttree->GetLeaf(name.c_str());

  if(!leaf)
  {
    std::stringstream ss;
    ss << "Variable named '" << name << "' not found !";
    throw GateKeyNotFoundInHeaderException(ss.str());
  }

  auto key = string(leaf->GetTypeName ());

//  cout << "t_index = " << t_index.name() << endl;
//  cout << "leaf = " << key << " recorded = " << m_tmapOfTindexToLongDefinition.at(t_index) << endl;
//  cout << "leaf GetLenStatic = " << leaf->GetLenStatic() << endl;
//  cout << "leaf GetLen = " << leaf->GetLen() << endl;

  if(key == "Char_t" and (t_index == typeid(string)  or t_index == typeid(char*) ))
    return;


  if( key !=  m_tmapOfTindexToLongDefinition.at(t_index) )
  {
    std::stringstream ss;
    ss << "type_index given to store '" << name <<"' has not the right type (given " << m_tmapOfTindexToLongDefinition.at(t_index) << " but need " << leaf->GetTypeName () << ")";
    throw GateTypeMismatchHeaderException(ss.str());
  }

}

void GateInputRootTreeFile::read_variable(const std::string &name, void *p, std::type_index t_index)
{
  this->check_existence_and_kind(name, t_index);
  m_ttree->SetBranchAddress(name.c_str(), p);
}

void GateInputRootTreeFile::read_variable(const std::string &name, char *p)
{
  this->check_existence_and_kind(name, typeid(char*));
  m_ttree->SetBranchAddress(name.c_str(), p);
}

void GateInputRootTreeFile::read_variable(const std::string &name, char *p, size_t)
{
  this->check_existence_and_kind(name, typeid(char*));
  this->read_variable(name, p);
}

void GateInputRootTreeFile::read_variable(const std::string &name, std::string *p)
{
  this->check_existence_and_kind(name, typeid(string));

  char* s = new char[1024];
  m_mapNewStringToRefString.emplace(s, p);
  m_ttree->SetBranchAddress(name.c_str(), s);
}

void GateInputRootTreeFile::read_header()
{
//    cout << "read tree name = " << m_nameOfTree << " from file " << endl;
    m_file->GetObject(m_nameOfTree.c_str(), m_ttree);
    assert(m_ttree);
    m_current_entry = 0;
    m_read_header_called = true;
}
void GateInputRootTreeFile::read_next_entrie()
{
    m_ttree->GetEntry(m_current_entry);

    for(auto ss: m_mapNewStringToRefString)
    {
        ss.second->assign((ss.first));
    }

    m_current_entry++;
}
bool GateInputRootTreeFile::data_to_read()
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
  return m_current_entry < m_ttree->GetEntries();
}



GateInputRootTreeFile::~GateInputRootTreeFile()
{
    for(auto ss: m_mapNewStringToRefString)
    {
        delete[](ss.first);
    }
}


void GateOutputRootTreeFile::set_tree_name(const std::string &name)
{
  GateOutputTreeFile::set_tree_name(name);
  m_ttree->SetName(name.c_str());
}

void GateInputRootTreeFile::set_tree_name(const std::string &name)
{
  GateInputTreeFile::set_tree_name(name);
}

GateInputRootTreeFile::GateInputRootTreeFile() : GateRootTree(), m_read_header_called(false)
{}



bool GateInputRootTreeFile::has_variable(const std::string &name)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  return m_ttree->GetListOfBranches()->FindObject(name.c_str());
}

type_index GateInputRootTreeFile::get_type_of_variable(const std::string &name)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  auto leaf = m_ttree->GetLeaf(name.c_str());
  auto key = string(leaf->GetTypeName ());

  if(key == "Char_t")
    throw GateNoTypeInHeaderException("type of string, char, char* can not be retrieved. No distinction are mage in ROOT file");

  return m_tmapOfLongDefinitionToIndex.at(key);
}


uint64_t GateInputRootTreeFile::nb_elements()
{
  assert(m_ttree);
  return m_ttree->GetEntries();
}

void GateInputRootTreeFile::read_entrie(const uint64_t &i)
{
  m_ttree->GetEntry(i);
  for(auto ss: m_mapNewStringToRefString)
  {
    ss.second->assign((ss.first));
  }
  m_current_entry = i+1;
}


bool GateOutputRootTreeFile::s_registered =
    GateOutputTreeFileFactory::_register(GateOutputRootTreeFile::_get_factory_name(), &GateOutputRootTreeFile::_create_method<GateOutputRootTreeFile>);

bool GateInputRootTreeFile::s_registered =
    GateInputTreeFileFactory::_register(GateOutputRootTreeFile::_get_factory_name(), &GateInputRootTreeFile::_create_method<GateInputRootTreeFile>);












