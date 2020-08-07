//
// Created by mdupont on 12/03/18.
//

#include "GateTreeFileManager.hh"


#include <iostream>
#include <memory>
#include <iostream>
#include <exception>
#include <sstream>
#include <algorithm>
#include <functional>

#include "GateFileExceptions.hh"


using namespace std;


//map<const string, const string> OutputTreeFileFactory::s_methods;


bool GateOutputTreeFileFactory::_register(const string name,
                                        TCreateOutputTreeFileMethod& funcCreate)
{
    auto it = get_methods_map().find(name);
    if(it != get_methods_map().end())
    {
      throw GateKeyAlreadyExistsException("nooooooooo");
    }

    get_methods_map().emplace(name, funcCreate);
    return true;
}

std::unique_ptr<GateOutputTreeFile>
GateOutputTreeFileFactory::_create(const string& name)
{
  auto it = get_methods_map().find(name);
  if(it != get_methods_map().end())
  {
    return it->second(); // call the createFunc
  }

  return nullptr;
}




GateOutputTreeFileManager::GateOutputTreeFileManager()
{
  m_nameOfTree = GateTree::default_tree_name();
}

GateOutputTreeFileManager::GateOutputTreeFileManager(GateOutputTreeFileManager &&m) :
m_listOfTreeFile(move(m.m_listOfTreeFile)),
m_nameOfTree(move(m.m_nameOfTree))
{}


void GateOutputTreeFileManager::write_variable(const std::string &name, const std::string *p, size_t nb_char)
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->write_variable(name, p, nb_char);
  }
}

void GateOutputTreeFileManager::write_variable(const std::string &name, const char *p, size_t nb_char)
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->write_variable(name, p, nb_char);
  }
}

void GateOutputTreeFileManager::write_variable(const std::string &name, const int *p, size_t sizeArray)
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->write_variable(name, p, sizeArray);
  }
}

void GateOutputTreeFileManager::write()
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->write();
  }
}

void GateOutputTreeFileManager::close()
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->close();
  }
}

void GateOutputTreeFileManager::fill()
{
  for(auto& f : m_listOfTreeFile)
  {
    f->fill();
  }
}

void GateOutputTreeFileManager::write_header()
{
  for(auto&& f : m_listOfTreeFile)
  {
    f->set_tree_name(m_nameOfTree);
    f->write_header();
  }
}

std::unique_ptr<GateOutputTreeFile> GateOutputTreeFileManager::add_file(const std::string &file_path, const std::string &kind)
{

  auto h = GateOutputTreeFileFactory::_create(kind);

  if(!h)
  {
    std::stringstream ss;
    ss << "Error do not know type '"  << kind; ;
    throw std::runtime_error(ss.str());
  }

  h->open(file_path);
  m_listOfTreeFile.push_back(move(h));
  return h;
}

GateOutputTreeFileManager::~GateOutputTreeFileManager()
{

}

void GateOutputTreeFileManager::set_tree_name(const std::string &name)
{
  m_nameOfTree = name;
}

bool GateInputTreeFileFactory::_register(const std::string name, TCreateInputTreeFileMethod &funcCreate)
{

  auto it = get_methods_map().find(name);
  if(it != get_methods_map().end())
  {
    throw GateKeyAlreadyExistsException("nooooooooo");
  }

  get_methods_map().emplace(name, funcCreate);
  return true;
}

std::unique_ptr<GateInputTreeFile> GateInputTreeFileFactory::_create(const std::string &name)
{
  auto it = get_methods_map().find(name);
  if(it != get_methods_map().end())
  {
    return it->second(); // call the createFunc
  }

  return nullptr;
}



GateInputTreeFileManager::GateInputTreeFileManager()
{
  m_nameOfTree = GateTree::default_tree_name();
}

std::unique_ptr<GateInputTreeFile> GateInputTreeFileManager::set_file(const std::string &file_path, const std::string &kind)
{
  auto h = GateInputTreeFileFactory::_create(kind);

  if(!h)
  {
    std::stringstream ss;
    ss << "Error do not know type '"  << kind; ;
    throw GateUnknownKindManagerException(ss.str());
  }

  h->open(file_path);
  m_inputtreefile = move(h);
  return h;
}

bool GateInputTreeFileManager::data_to_read()
{
  return m_inputtreefile->data_to_read();
}

void GateInputTreeFileManager::close()
{
  if(m_inputtreefile)
    m_inputtreefile->close();
}

void GateInputTreeFileManager::read_next_entrie()
{
  m_inputtreefile->read_next_entrie();
}

void GateInputTreeFileManager::read_header()
{
  m_inputtreefile->set_tree_name(m_nameOfTree);
  m_inputtreefile->read_header();
}

void GateInputTreeFileManager::read_variable(const std::string &name, void *p, std::type_index t_index)
{
  m_inputtreefile->read_variable(name, p, t_index);
}

void GateInputTreeFileManager::read_variable(const std::string &name, char *p)
{
  m_inputtreefile->read_variable(name, p);
}

void GateInputTreeFileManager::read_variable(const std::string &name, char *p, size_t nb_char)
{
  m_inputtreefile->read_variable(name, p, nb_char);
}

void GateInputTreeFileManager::read_variable(const std::string &name, std::string *p)
{
  m_inputtreefile->read_variable(name, p);
}


void GateInputTreeFileManager::set_tree_name(const std::string &name)
{
  m_nameOfTree = name;
}

bool GateInputTreeFileManager::has_variable(const std::string &name)
{
  return m_inputtreefile->has_variable(name);
}

std::type_index GateInputTreeFileManager::get_type_of_variable(const std::string &name)
{
  return m_inputtreefile->get_type_of_variable(name);
}

uint64_t GateInputTreeFileManager::nb_elements()
{
  return m_inputtreefile->nb_elements();
}

void GateInputTreeFileManager::read_entrie(const uint64_t &i)
{
  m_inputtreefile->read_entrie(i);
}


//std::unique_ptr<InputTreeFile> InputTreeFileManager::get_file(const std::string &file_path, const std::string &kind)
//{
//
//    auto h = InputTreeFileFactory::_create(kind);
//
//    if(!h)
//    {
//        std::stringstream ss;
//        ss << "Error do not know type '"  << kind; ;
//        throw UnknownKindManagerException(ss.str());
//    }
//
//    h->open(file_path);
////    m_inputtreefile = move(h);
//    return h;
//}
std::unique_ptr<GateInputTreeFile> GateInputTreeFileChain::add_file(const std::string &file_path, const std::string &kind)
{
  auto h = GateInputTreeFileFactory::_create(kind);

  if(!h)
  {
    std::stringstream ss;
    ss << "Error do not know type '"  << kind; ;
    throw std::runtime_error(ss.str());
  }

  h->open(file_path);
  m_listOfTreeFile.push_back(move(h));
  return h;
}

bool GateInputTreeFileChain::data_to_read()
{
  for(auto &f: m_listOfTreeFile)
  {
    if(f->data_to_read())
      return true;
  }
  return false;
}

void GateInputTreeFileChain::read_entrie()
{
  for(auto &f: m_listOfTreeFile)
  {
    if(f->data_to_read())
    {
      f->read_next_entrie();
      return;
    }
  }

}

void GateInputTreeFileChain::read_header()
{
  for(auto &f: m_listOfTreeFile)
  {
    f->set_tree_name(m_nameOfTree);
    f->read_header();
  }
}

void GateInputTreeFileChain::read_variable(const std::string &name, char *p)
{
  for(auto &f: m_listOfTreeFile)
  {
    f->read_variable(name, p);
  }

}

void GateInputTreeFileChain::read_variable(const std::string &name, std::string *p)
{
  for(auto &f: m_listOfTreeFile)
  {
    f->read_variable(name, p);
  }
}

void GateInputTreeFileChain::read_variable(const std::string &name, char *p, size_t nb_char)
{
  for(auto &f: m_listOfTreeFile)
  {
    f->read_variable(name, p, nb_char);
  }
}

void GateInputTreeFileChain::read_variable(const std::string &name, void *p, std::type_index t_index)
{
  for(auto &f: m_listOfTreeFile)
    f->read_variable(name, p, t_index);

}

void GateInputTreeFileChain::close()
{
  for(auto &f: m_listOfTreeFile)
    f->close();
}

void GateInputTreeFileChain::set_tree_name(const std::string &name)
{
  m_nameOfTree = name;
}

bool GateInputTreeFileChain::has_variable(const std::string &name)
{
  for(auto &f: m_listOfTreeFile)
  {
    if(!f->has_variable(name))
      return false;
  }
  return true;
}

std::type_index GateInputTreeFileChain::get_type_of_variable(const std::string &name)
{
  vector<std::type_index> v;
  for(auto &f: m_listOfTreeFile)
  {
    try
    {
      v.push_back(f->get_type_of_variable(name));
      cout << "file = " << f->path()  << " t = " <<  f->get_type_of_variable(name).name() << endl;
    }
    catch (const GateNoTypeInHeaderException&)
    {

    }
  }

  if(v.empty())
    throw GateNoTypeInHeaderException("Provided files not able to provide type in data");

  if ( std::adjacent_find( v.begin(), v.end(), std::not_equal_to<std::type_index>() ) == v.end() )
  {
    // All elements are equals
    return v.at(0);
  }
  else
    throw runtime_error("Differents type index !");



}

uint64_t GateInputTreeFileChain::nb_elements()
{
  uint64_t n = 0;
  for(auto &f: m_listOfTreeFile)
  {
    n += f->nb_elements();
  }
  return n;
}

GateInputTreeFileChain::GateInputTreeFileChain()
{
  m_nameOfTree = GateTree::default_tree_name();
}

void GateInputTreeFileChain::read_entrie(const uint64_t &i)
{
  uint64_t seek = i;

  for(auto &f: m_listOfTreeFile)
  {
    if(seek < f->nb_elements())
    {
      f->read_entrie(seek);
      return;
    }
    seek -= f->nb_elements();
  }
}
