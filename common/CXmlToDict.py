# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from lxml import objectify as xml_objectify

import pathlib
from lxml import etree
from collections import OrderedDict


###########################
# xml to dict
###########################
class XmlConvertor:
    def xml2dict(self, fpath):
        """"""
        with open(fpath, "r") as f:
            # tree type:lml.etree._ElementTree
            tree = etree.parse(f)
            # tree.getroot() type:tree.getroot(), xml.etree._Element
            return self.tree2dict(tree.getroot())

    def tree2dict(self, node):
        """"""
        subdict = OrderedDict({})
        # iterate over the children of this element--tree.getroot
        for e in node.iterchildren():
            d = self.tree2dict(e)
            for k in d.keys():
                # handle duplicated tags
                if k in subdict:
                    v = subdict[k]
                    # use append to assert exception
                    try:
                        v.append(d[k])
                        subdict.update({k: v})
                    except AttributeError:
                        subdict.update({k: [v, d[k]]})
                else:
                    subdict.update(d)
        if subdict:
            return {node.tag: subdict}
        else:
            return {node.tag: node.text}

    def xml_to_dict(self, xml_str):
        """ Convert xml to dict, using lxml v3.4.2 xml processing library """

        def xml_to_dict_recursion(xml_object):
            dict_object = xml_object.__dict__
            if not dict_object:
                return xml_object
            for key, value in dict_object.items():
                dict_object[key] = xml_to_dict_recursion(value)
            return dict_object

        return xml_to_dict_recursion(xml_objectify.fromstring(xml_str))

    def Get_Dict(self,filename):
        data = ""
        filename = os.path.dirname(
            pathlib.Path(__file__).parent.resolve()) + filename
        with open(filename, "r") as xmlfile:
            for line in xmlfile.readlines():
                data = data + line
        Dict = self.xml_to_dict(data)
        return Dict

    def get_list(self):
        dic = self.Get_Dict('/Configuration/ChartConfig/ExtraAttrXaxis.xml')
        List = list(dic.keys())
        return List
