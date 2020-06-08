//
// Created by mdupont on 12/03/18.
//

#pragma once

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <fstream>


#include "GateFile.hh"

class GateData
{
public:
  GateData(const void * pointer_to_data,
       const std::string _name,
       const std::type_index type_index

  ) : m_pointer_to_data(pointer_to_data),
      m_name(_name),
      m_type_index(type_index)
  {

  }

  const std::string &name() const
  {
    return m_name;
  }

  const void *m_pointer_to_data;
  const std::string m_name;
  std::type_index m_type_index;
};



class GateTree
{
public:
  static std::string default_tree_name();
  GateTree();
  template <class T>
  static std::unique_ptr<T> _create_method() {
    return std::unique_ptr<T>(new T());
  }

protected:
  virtual void register_variable(const std::string &name, const void *p, std::type_index t_index) = 0;
  virtual void register_variable(const std::string &name, const std::string *p, size_t nb_char) = 0;
  virtual void register_variable(const std::string &name, const char *p, size_t nb_char) = 0;
  virtual void register_variable(const std::string &name, const int *p, size_t n) = 0;


private:
  template<typename T>
  void add_size()
  {
    m_tmapOfSize[typeid(T)] = sizeof(T);
  }

  template<typename T>
  void add_name(const std::string &name)
  {
//    m_tmapOfName[typeid(T)] = name;
    m_tmapOfName.emplace(typeid(T), name);
  }




protected:
  const std::string type_to_name(std::type_index t_index);
  std::unordered_map<std::type_index, std::size_t> m_tmapOfSize;
  std::unordered_map<std::type_index, std::string> m_tmapOfName;
};



class GateOutputTreeFile : public GateFile
{
public:
  GateOutputTreeFile();

  virtual void open(const std::string& s) = 0;
  virtual void write_header() = 0;
  virtual void write() = 0;
  virtual void fill() = 0;


  virtual void write_variable(const std::string &name, const void *p, std::type_index t_index) = 0;
  virtual void write_variable(const std::string &name, const std::string *p, size_t nb_char) = 0;
  virtual void write_variable(const std::string &name, const char *p, size_t nb_char) = 0;
  virtual void write_variable(const std::string &name, const int  *p, size_t n) = 0;
  virtual void set_tree_name(const std::string &name) ;
  virtual ~GateOutputTreeFile();

protected:
  std::string m_nameOfTree;
};


class GateInputTreeFile : public GateFile
{
public:

  GateInputTreeFile();
  virtual ~GateInputTreeFile() = default;
  virtual void open(const std::string& s) = 0;
  virtual void read_header() = 0;
  virtual void read_next_entrie() = 0;
  virtual void read_entrie(const uint64_t& i) = 0;
  virtual bool data_to_read() = 0;
  virtual void set_tree_name(const std::string &name) ;
  virtual void read_variable(const std::string &name, void *p, std::type_index t_index) = 0;
  virtual void read_variable(const std::string &name, char* p);
  virtual void read_variable(const std::string &name, char* p, size_t nb_char);
  virtual void read_variable(const std::string &name, std::string *p);
  virtual bool has_variable(const std::string &name) = 0;
  virtual std::type_index get_type_of_variable(const std::string &name) = 0;
  virtual uint64_t nb_elements() = 0;


  template<typename T>
  void read_variable(const std::string &name, T *p)
  {
      read_variable(name, p, typeid(T));
  }

protected:
  std::string m_nameOfTree;
};

