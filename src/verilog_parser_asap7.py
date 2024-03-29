from cmath import sin
import os

from typing import List, Dict, Tuple, Optional
from xml.etree.ElementTree import PI
import pyverilog
from pyverilog.vparser.parser import parse
from pyverilog.vparser.parser import *
import re
import pickle
import networkx as nx
from time import time
import torch as th
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path
import dgl
import json
import cProfile

with open('../rawdata/cell_info_map2.json', 'r') as f:
    cell_info_map = json.load(f)

with open('../rawdata/early_lib.json', 'r') as f:
    cell_lib = json.load(f)

map_size_x, map_size_y = 128,128

def parse_arg(arg,port_info,father_wires):
    r"""

    parse the information of argument
        e.g., {a[3],b[10:2],1'b0,c}

    :param arg:
                the argument
    :param port_info: PortInfo
                the port that the argument belongs to
    :param father wires: dict{str:(int,int)}
                    {wire_name: (high bit, low bit)}
                    dictionary of wires in the father module
    :return:
    """

    if type(arg) == pyverilog.vparser.ast.Identifier:
        if father_wires.get(arg.name, None) is not None:
            wire_type,high_bit, low_bit = father_wires[arg.name]
        else:
            print(arg.name)
            assert False
        width = high_bit - low_bit + 1
        if width == 1:
            port_info.arg_list.append(arg.name)
        else:
            for i in range(high_bit, low_bit - 1, -1):
                port_info.arg_list.append("{}[{}]".format(arg, i))
    # const, e.g., 1'b0
    elif type(arg) == IntConst:
        port_info.arg_list.append(arg.value)
    # parselect, e.g., a[n1:n2]
    elif type(arg) == Partselect:
        arg_nm,high_bit,low_bit = arg.children()
        arg_nm = arg_nm.name
        # get the highest/lowest bit
        high_bit, low_bit = int(str(high_bit)),int(str(low_bit))
        if high_bit < low_bit:
            temp = high_bit
            high_bit = low_bit
            low_bit = temp
        # add the arg to arglist
        for i in range(high_bit,low_bit-1,-1):
            port_info.arg_list.append("{}[{}]".format(arg_nm,i))
    # pointer, e.g., a[n]
    elif type(arg) == Pointer:
        arg_nm, position = arg.children()
        arg_nm = arg_nm.name
        port_info.arg_list.append("{}[{}]".format(arg_nm,position))
    else:
        print(arg)
        assert False

class PortInfo:
    ptype:str
    portname: str
    arg_list: list
    def __init__(self, portname):
        self.ptype = None
        self.portname = portname
        self.arg_list = []

class PinInfo:
    cell_type: str
    pin_name: str
    net_name: str
    def __init__(self,cell_type, pin_name,net_name):
        self.cell_type = cell_type
        self.pin_name = pin_name
        self.net_name = net_name


def parse_pin(pin,pin_info_along_path):
    """

    given the full name of a pin, and a timing path that contain the pin,
    return the detailed information about the pin.
    The timing path is represented by a directory of the pin information along the path.

    :param pin: str
                full pin name
                    e.g., system/tile_prci_domain/tile_reset_domain/tile/core/ex_reg_rs_msb_0_reg[1]/CLK
    :param pin_info_along_path: {full_pin_name:[net_name, cell_type]}
                full_pin_names, net_name, cell_type all are of type str
                    e.g., 'system/.../fq/CLKGATE_RC_CG_HIER_INST552/RC_CGIC_INST/CLK': ['clock_clock', 'ICGx1_ASAP7_75t_SL']
    :return:
        pin_info: PinInfo
            this is a self-defined class that records the detailed information about a pin.
    """
    net_name, cell_type = pin_info_along_path[pin]

    pin_info = PinInfo(cell_type,pin,net_name)
    return pin_info

def update_netname(net_name,call_path,io2arg):

    """
    update the net name if it is connected to the outside netlist.
        # check if the net is the input/output of the current module
        #     if true, then update the net name

    :param net_name: str
                name of the net
    :param call_path: str
                the calling path of the module
                    e.g., ChipTop/.../
    :param io2arg: dict{str:(str,int)}
                {input/output net name: (port argument name, trace depth)}
                dictionary that maps the input/output ports to arguments in father/grandpa module
                    trace_depth=1: father ; trace_back=2: grandpa
    :return:
    """

    # check if the net is the input/output of the current module
    #     if true, then trace back along the call path,
    #       and update the net name with the argname from father module,
    #       then add the corresponding traced-back call path in the front.
    #     if false, then simplify add the call path in the front.
    #arg_name = net_name.replace('\\','')
    arg_name = net_name
    if io2arg is not None and \
            io2arg.get(net_name, None) is not None:
        _,arg_name, trace_depth = io2arg[net_name]
        for _ in range(trace_depth):
            if '/' in call_path:
                call_path = call_path[:call_path.rfind('/')]
            else:
                call_path = ''
    if call_path == '':
        net_name = arg_name
    else:
        net_name = '{}/{}'.format(call_path, arg_name)

    return net_name

def pin2bin(pin_x,pin_y,bin_size_x,bin_size_y):
    """

    map the pin position to bin position

    :param pin_x:
    :param pin_y:
    :param bin_size_x:
    :param bin_size_y:
    :return:
    """
    bin_x = min(max(int(pin_x/bin_size_x), 0), map_size_x-1)
    bin_y = min(max(int(pin_y/bin_size_y),0),map_size_y-1)

    return bin_x,bin_y



class PathInfo:
    start: str
    end: str
    is_critical:  bool
    required_time: int
    arrival_time: int
    path: List[str]
    nets: List[str]

    def __init__(self,start=None,end=None,path=None,nets=None,required_time=None,arrival_time=None):
        self.start = start
        self.end = end
        self.path = path
        self.nets = nets
        self.is_critical = False
        self.required_time = required_time
        self.arrival_time = arrival_time
        #self.mask = th.zeros((map_size_x,map_size_y))
class NetInfo:
    drive_cell:str
    net_name: str
    drive_pin: str
    sink_pins : List[str]
    total_output_cap:int
    def __init__(self,net_name):
        self.drive_cell = ''
        self.net_name = net_name
        self.drive_pin = ''
        self.sink_pins = []
        self.total_output_cap = 0

class Parser:
    def __init__(
        self, top_module: str, masking
    ):
        self.top_module = top_module
        self.masking = masking
        self.ast = None
        self.graph = None
        self.nets = {}
        self.module_wires_map = {}
        self.module_IO2arg_map = {}
        self.module_IO2pin_map = {}
        self.timing_paths = {}
        # self.critical_ends = {}
        self.num_paths = 0
        self.net_bbox_map = {}
        self.equal_wire_map = {}
        self.cell_type_count = {}

        self.pin2delay = {}
        self.pin2trans = {}
        self.endpoints = []

    def is_output_port(self, cell:str,port: str) -> bool:
        return cell_lib[cell]['pin_info'][port]['direction'] == 'output'
        #return port in ("Y", "S", "SO", "CO", "Q", "QN","io_q","H","L",
        #                "GCLK",'SN',"O",'CON')

    def parse_pin_locations(self,file):
        res = {}
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('==='):
                continue
            pin_name, coord_x, coord_y = line.split(' ')
            pin_name = pin_name.replace('\[', '[').replace('\]', ']').replace('\\','')
            coord_y = coord_y.replace('\n', '')
            coord_x, coord_y = float(coord_x), float(coord_y)
            #bin_x, bin_y = coord_x / self.bin_size_x, coord_y / self.bin_size_y
            # we multiply bin_size by 4 here is because the output feature map size is 1/4 of input map
            bin_x, bin_y = pin2bin(coord_x, coord_y, int(512/map_size_x), int(512/map_size_y))
            #bin_x, bin_y = int(coord_x/4), coord_y/4
            res[pin_name] = (coord_x, coord_y, bin_x, bin_y)

        return res

    def parse_path(self,path_text):
        """

        given the path text extracted from the timing report, parse the information about the
        pins along the path, including pin name, connected net and cell type.

        :param path_text: str
                    the extracted path text
        :return:
            path_info_along_path: {full_pin_name:[net_name, cell_type]}
                    full_pin_names, net_name, cell_type all are of type str
                    e.g., 'system/.../fq/CLKGATE_RC_CG_HIER_INST552/RC_CGIC_INST/CLK':
                                ['clock_clock', 'ICGx1_ASAP7_75t_SL']
        """
        lines = path_text.split('\n')

        path,pins = [],[]
        nets = set()
        startpoint,endpoint = None,None
        required_time, arrival_time =0,0
        flag_point, flag_start = False,False
        for line in lines:
            # parse each line
            if 'Startpoint' in line:
                startpoint = line.split(' ')[-1]
                #    assert False, '{}'.format(startpoint)
            elif 'Endpoint' in line:
                endpoint = line.split(' ')[-1]
            elif 'Required Time' in line:
                required_time = float(line.split(' ')[-1])
            elif 'Data Path:' in line:
                arrival_time = float(line.split(' ')[-1])
            elif 'Timing Point' in line:
                flag_point = True

            #print(line,flag_start)
            #if line.startswith('#') and flag_start:
            #    break
            if line.startswith('#') or not flag_point:
                continue
            context = [c for c in line.split(' ') if len(c)!=0]
            #print(context)
            pin, flag, arc, edge,cell,fanout, trans,delay,arrival= context
            if cell=='(net)':
                if flag_start: nets.add(pin)
            elif cell=='(arrival)':
                continue
            elif  '->' not in arc:
                path.append(pin)
                break
            #elif  '->' not in arc:
            #    if pin == startpoint:
            #        flag_start = True
            #    if flag_start:
            #        path.append(pin)
            else:
                self.pin2delay[pin] = float(delay)
                self.pin2trans[pin] = float(trans)
                drive_port, sink_port = arc.split('->')
                cell_name = pin[:pin.rfind('/')]
                drive_pin = cell_name + '/' + drive_port
                sink_pin = cell_name + '/' + sink_port
                if drive_pin == startpoint:
                    flag_start = True
                    path.append(sink_pin)
                    startpoint = sink_pin
                    continue
                if flag_start:
                    path.append(drive_pin)
                    path.append(sink_pin)

            #print(pin,'a')

        # deal with the last pin

        #print(path,nets,required_time,arrival_time)
        return startpoint,endpoint,path,nets,required_time,arrival_time

    def parse_postoptReport(self, fname):
        r"""

        parse the post-optimization timing report to find the critical endpoints

        :param fname: str
                    the path of the report file
        :return:
        """

        print('###  parsing the post-opt timing report file...')
        if not os.path.exists(fname):
            print('\tError: report file doest not exist!')
            exit()
        with open(fname, 'r') as f:
            text = f.read()
        print('\treport file is read.')

        # split the report into blocks that each block contain 1 timing path
        # critical_ends = {}
        blocks = text.split('Check with')
        path_state = blocks[0].split('\n')[-1].split(' ')[2]
        blocks = blocks[1:]
        all_paths = {}
        post_criticals = []
        # parse all the timing paths in the report, get the following information:
        #       start point, end point, path, is_criticalcal endpoints
        for i, block in enumerate(blocks):
            # parse the information along the path
            startpoint,endpoint,path,nets,required_time,arrival_time = self.parse_path(block)
            self.endpoints.append(endpoint)
            path_info = PathInfo(end=endpoint,required_time=required_time,arrival_time=arrival_time)
            if path_state == 'VIOLATED':
                path_info.is_critical = True
                #critical_ends[endpoint] = True
                post_criticals.append(i)
            elif path_state != 'MET':
                assert False, 'wrong state {} for path {}'.format(path_state,i+1)
            if i != len(blocks) - 1:
                path_state = block.split('\n')[-1].split(' ')[2]

            self.timing_paths[endpoint] = path_info

            all_paths[i] = path

        #self.critical_ends = critical_ends

        with open(os.path.join(self.data_path,'post_paths.pkl'),'wb') as f:
            pickle.dump((all_paths,post_criticals),f)
        print('### Parsing is done.')
        print('#paths: {}, #critical: {}'.format(len(all_paths),len(post_criticals)))
        #return critical_ends

    def parse_preoptReport(self,fname):
        r"""

        parse the post-optimization timing report to find the start and end node of the timing paths

        :param fname: str
                    the path of the report file
        :return:
            timing_paths: List[PathInfo]
                    the detailed information about the timing paths
        """

        # assert len(self.critical_ends)!=0, 'No critical endpoints! ' \
        #                                    'Please call parse_postoptReport before parse_preoptReport'

        # paths = {}
        print('###  parsing the post-opt timing report file...')
        if not os.path.exists(fname):
            print('\tError: report file doest not exist!')
            exit()
        with open(fname,'r') as f:
            text = f.read()
        print('\treport file is read.')

        # split the report into blocks that each block contain 1 timing path
        blocks  = text.split('Check with')
        path_state = blocks[0].split('\n')[-1].split(' ')[2]
        blocks = blocks[1:]
        # parse all the timing paths in the report, get the following information:
        #       start point, end point, path, is_critical
        print('\t-----parsing the paths one by one...')

        all_paths = {}
        pre_criticals = []
        for i,block in enumerate(blocks):
            # get the path text (list of pins along the path)

            # parse the pins' information along the path
            startpoint,endpoint,path,nets,required_time,arrival_time= self.parse_path(block)

            if path_state == 'VIOLATED':
                pre_criticals.append(i)
            if i != len(blocks) - 1:
                path_state = block.split('\n')[-1].split(' ')[2]
            # get the path info, and add to result

            all_paths[i] = path

            self.timing_paths[endpoint].start = startpoint
            self.timing_paths[endpoint].path = path
            self.timing_paths[endpoint].nets = nets
            # path_info = PathInfo(startpoint, endpoint, path,nets)
            # paths[endpoint] = path
            # if self.critical_ends.get(endpoint,False):
            #     path_info.is_critical = True
            # timing_paths.append(path_info)

        # with open('../data/paths.pkl','wb') as f:
        #     pickle.dump(paths,f)

        with open(os.path.join(self.data_path,'pre_paths.pkl'),'wb') as f:
            pickle.dump((all_paths,pre_criticals),f)
        #print the paths
        num_critical = 0
        #print('\t-----Found critical timing paths: ')
        for i,path_info  in enumerate(self.timing_paths.values()):
            if False:
                print('\tpath {},'.format(i+1),'critcal:',True )
                print('\t\t required {}, arrival {}'.format(path_info.required_time,path_info.arrival_time))
                print('\t\t start: ',path_info.start)
                print('\t\t end: {} '.format(path_info.end))
                print('\t\t path length: ',len(path_info.path))
                slack = path_info.required_time - path_info.arrival_time
                #assert slack<0
                num_critical +=1
        #print(num_critical)

        self.timing_paths = list(self.timing_paths.values())
        self.num_paths = len(self.timing_paths)

        return self.timing_paths


    def parse_wires(self,module):
        """

        parse the wires (input, ouput, wire) of the module.
              e.g.,  wire a[31:2];
        record each declaration and its range (highest bit and lowest bit)

        :param module: Module

        :return:
            wires: dict{str:(str,int,int)}
                        {wire_name: (wire_type,high bit, low bit)}
                        dictionary of wires and their range
        """

        wires = {}
        equal_wire_map = {}
        #parse all the declarations
        #   e.g.,
        #       Decl:  (at 31)
        #           Input: clock, False (at 31)
        #           Input: reset, False (at 31)
        #           Input: io_d, False (at 31)
        #
        #       Decl:  (at 36)
        #           Wire: a1, False (at 36)
        #               Dimensions:  (at 36)
        #                   Length:  (at 36)
        #                       IntConst: 64 (at 36)
        #                       IntConst: 0 (at 36)

        for sentence in module.children():
            if type(sentence) == Decl:
                for decl in sentence.children():
                    name = decl.name
                    # parse the highest/lowest bit
                    if decl.dimensions is None and decl.width is None:
                        high_bit, low_bit = 0, 0
                    elif decl.dimensions is not None:
                        length = decl.dimensions.lengths[0]
                        high_bit, low_bit = length.children()
                        high_bit, low_bit = int(high_bit.value), int(low_bit.value)
                        if high_bit < low_bit:
                            temp = high_bit
                            high_bit = low_bit
                            low_bit = temp
                    elif decl.width is not None:
                        high_bit, low_bit = decl.width.children()
                        high_bit, low_bit = int(high_bit.value), int(low_bit.value)
                        if high_bit < low_bit:
                            temp = high_bit
                            high_bit = low_bit
                            low_bit = temp
                    # save the highest/lowest bit of each io / wire
                    if type(decl) == Input:
                        wire_type = 'i'
                    elif type(decl) == Output:
                        wire_type = 'o'
                    else:
                        wire_type = 'w'
                    if wires.get(name,None) is None:
                        wires[name] = (wire_type,high_bit, low_bit)
            elif type(sentence) == pyverilog.vparser.ast.Wire:
                name = sentence.name
                wire_type = 'w'
                wires[name] = (wire_type,0, 0)
            # record the equavalent wires (assign)
            elif type(sentence) == pyverilog.vparser.ast.Assign:

                lf_value = sentence.children()[0].var
                if type(lf_value) == Identifier:
                    lf_value = '{}'.format(lf_value)
                elif type(lf_value) == Pointer:
                    lf_arg,lf_position =lf_value.children()
                    lf_value = '{}[{}]'.format(lf_arg,lf_position)
                #print(lf_value)
                rt_value = sentence.children()[1].var
                if type(rt_value) == Identifier:
                    rt_value = '{}'.format(rt_value)
                elif type(rt_value) == Pointer:
                    rt_arg, rt_position = rt_value.children()
                    rt_value = '{}[{}]'.format(rt_arg, rt_position)
                #print(rt_value)
                equal_wire_map[lf_value] = rt_value

        return wires,equal_wire_map

    def parse_io2arg(self,portlist,wires,father_wires,father_io2arg):
        """

        parse the portlist of the module, to find the mapping from inputs/outputs of the module
        to their arguments (nets from outside),
        so that we trace back along the call path, and replace the input/outputs with the nets from outside

        :param portlist: PortList
                    list of ports of the current module
        :param wires: dict{str:(str, int,int)}
                    {wire_name: (wire_type,high bit, low bit)}
                    dictionary of wires in the current module
                    used to get the range of input/output
        :param father_wires: dict{str:(str, int,int)}
                    {wire_name: (wire_type, high bit, low bit)}
                    dictionary of wires in the father module
                    used to update the net names connected to father module
        :param father_io2arg: dict{str:(str,str,int)}
                    {input/output net name: (wire_type,port argument name, trace depth)}
                    dictionary that maps the input/output ports of father module to arguments in grandpa module

        :return:
            io2arg: dict{str:(str,str,int)}
                     {input/output net name: (wire_type, port argument name, trace depth)}
                    dictionary that maps the input/output ports to arguments in father module

        """

        io2arg = {}
        for p in portlist:
            port_info = self.parse_moduleport(p, father_wires)
            portname = port_info.portname
            wire_type,high_bit, low_bit = wires[portname]
            width = high_bit - low_bit + 1
            if width == 1:
                arg_name = port_info.arg_list[0]
                # if the argument is not the input/output of the father module
                # then the updating is accomplished,
                #       trace detph = 1
                io2arg[portname] = (wire_type,arg_name, 1)
                # if the argument is also the input/output of the father module,
                # we need to further trace back along the call path by increasing the trace depth
                #       trace_depth +=1
                if father_io2arg is not None and \
                        father_io2arg.get(arg_name, None) is not None:
                    io2arg[portname] = \
                        (wire_type, father_io2arg[arg_name][1],father_io2arg[arg_name][2]+1)
            else:
                indx2 = 0
                for i in range(high_bit, low_bit - 1, -1):
                    arg_name = port_info.arg_list[indx2]
                    io2arg['{}[{}]'.format(portname, i)] = (wire_type,arg_name, 1)
                    if father_io2arg is not None and \
                            father_io2arg.get(arg_name, None) is not None:
                        io2arg['{}[{}]'.format(portname, i)] = \
                            (wire_type, father_io2arg[arg_name][1],father_io2arg[arg_name][2]+1)
                    indx2 += 1

        return io2arg


    def parse_module(self,module_name,instance_name,portlist,
                     call_path):
        """

        given an instance of a module, parse the module into a sub-graph,
        where the input/output ports (that are connected to outside)
        are replaced with nets from father module.
                e.g., Test t1 (.A(a), .B(b));

        :param top_module_name: str
                    the top module name
                        e.g., ChipTop
        :param module_name: str
                    the module name
                        e.g., Test
        :param instance_name: str
                    the instance name
                        e.g., t1
        :param portlist: PortList
                    list of the instance's ports
        :param father_wires: dict{str:(str, int,int)}
                    {wire_name: (wire_type, high bit, low bit)}
                    dictionary of wires in the father module
                    used to update the net names connected to father module
        :param father_io2arg: dict{str:(str,str,int)}
                    {input/output net name: (wire_type,port argument name, trace depth)}
                    dictionary that maps the input/output ports to arguments in father/grandpa module
                        trace_depth=1: father ; trace_back=2: grandpa
        :param call_path: str
                    the calling path of the module
                            e.g., ChipTop/.../
        :return:
            nodes : List[Tuple[str, Dict[str, str]]
                        list of the nodes, where each node correspond to a pin
                        Dict gives the feature of the pin
            edges: List[Tuple[str, str, Dict[str, bool]]
                        list of the edges, where each edge corresponds to a cell-edge / net-edge
                        Dict gives the feature of the cell/net
        """
        # we treat each as pin as a node, and the cell/net as edge
        nodes: List[Tuple[str, Dict[str, str]]
        ] = []  # a list of (node, {})
        edges: List[
            Tuple[str, str, Dict[str, bool]]
        ] = []  # a list of (src, dst, {})

        # first find the module
        module = None
        for m in self.ast.description.definitions:
            if m.name == module_name:
                module = m
                break
        assert module is not None, 'Target module {} is not found!'.format(module_name)

        # parse the wires (input, ouput, wire) of the module.
        #       e.g.,  wire a[31:2];
        # record each declaration and its range (highest bit and lowest bit)
        wires,equal_wire_map = self.parse_wires(module)


        # parse the portlist of the module, to find the mapping from
        # inputs/outputs of the module to their arguments (nets from outside),
        # so that we replace the input/outputs with the nets from the outside
        if module_name == self.top_module:
            io2arg = None
            child_call_path = ''
        else:
            father_wires = self.module_wires_map[call_path]
            father_io2arg = self.module_IO2arg_map[call_path]
            io2arg = self.parse_io2arg(portlist,wires,father_wires,father_io2arg)
            if call_path == '':
                child_call_path = instance_name
            else:
                child_call_path = '{}/{}'.format(call_path,instance_name)

        for w1,w2 in equal_wire_map.items():
            w1 = update_netname(w1,call_path,io2arg)
            w2 = update_netname(w2,call_path,io2arg)
            self.equal_wire_map[w1] = w2
        self.module_wires_map[child_call_path] = wires
        self.module_IO2arg_map[child_call_path] = io2arg
        # parse the instances line by line
        #       e.g., Test1 t1(.A(a), .B(b));
        for item in module.items:
            if type(item) != InstanceList:
                continue
            item = item.instances[0]
            item_cell = item.module     #cell/module name, e.g, Test1
            item_name = item.name       # instance name, e.g., t1
            item_portlist = item.portlist    # portlist of the instance

            # if this is a instance of a module, then recursively call the 'parse_module'
            # else call 'parse_cell'
            if item_cell in self.modulelist:
                sub_nodes,sub_edges = self.parse_module(item_cell, item_name, item_portlist,
                                                        child_call_path)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
            else:
                if item_cell.startswith('SRAM'):
                    sub_nodes,sub_edges = self.parse_RAM(item_cell, item_name,item_portlist,wires,io2arg,
                                child_call_path)
                else:
                    sub_nodes,sub_edges = self.parse_cell(item_cell, item_name,item_portlist,io2arg,
                                    child_call_path)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)

        # print the detailed information of a module
        # if module_name == 'data_arrays_0_0_ext':
        #     print('\t io2arg:')
        #     for  key,value in io2arg.items():
        #         print('\t\t {} : {}'.format(key,value))
        #     print('\n\t nodes:')
        #     for node in nodes:
        #         print('\t\t',node)
        #     print('\n\t edges:')
        #     for edge in edges:
        #         print('\t\t', edge)
        return nodes,edges

    def parse_RAM(self,cell_name,instance_name,portlist,wires,io2arg,call_path):
        """

                given an instance of a SRAM, parse the SRAM into a sub-graph,
                where the input/output ports (that are connected to outside)
                are replaced with nets from father module.
                Note: for ram,we did not link the input pins and output pins,
                      so there will be no cel-edge!

                :param cell_name: str
                :param portlist: PortList
                :param io2arg: dict{str:(str,int)}
                :param call_path: str

                :return:
                    nodes : List[Tuple[str, Dict[str, str]]
                    edges: []

        """

        num_fanin = len(portlist)-2 if cell_name.startswith('HA') else len(portlist)-1
        fanins,fanouts = [],[]
        sub_nodes = []
        sub_edges = []
        for p in portlist:
            # sram is special because it is a macro cell, that is to
            # say, it should be treated as a model;
            # another thing is that we do not connect ram's input pin with its output pin
            port_info = self.parse_moduleport(p, wires)
            portname = port_info.portname
            # parse the pins (ports) of sram
            width = len(port_info.arg_list)
            for i,netname in enumerate(port_info.arg_list):
                netname = update_netname(netname,call_path,io2arg)
                netname = netname.replace('\\','')

                if call_path =='':
                    pinname = '{}/{}[{}]'.format(instance_name, port_info.portname, width-1-i) if width>1 \
                                else '{}/{}'.format(instance_name, port_info.portname)
                else:
                    pinname = '{}/{}/{}[{}]'.format(call_path,instance_name,port_info.portname,width-1-i) if width>1 \
                                else '{}/{}/{}'.format(call_path,instance_name,port_info.portname)

                # get the location of the current cell instance
                if self.pin_loc_map.get(pinname, None) is None:
                    assert False, 'pin with no location: {}'.format(pinname)
                pin_location = self.pin_loc_map[pinname]

                if self.nets.get(netname, None) is None:
                    self.nets[netname] = NetInfo(netname)
                if self.is_output_port(cell_name,portname):
                    pin_type = 'drive'
                    self.nets[netname].drive_pin = pinname
                    self.nets[netname].drive_cell = cell_name
                    fanouts.append([pinname,portname])
                elif 'CLK' in portname:
                    pin_type = 'sink'
                    self.nets[netname].sink_pins.append(pinname)
                    cap = float(cell_info_map[cell_name]['pin_info'][portname]['capacitance'])
                    self.nets[netname].total_output_cap += cap
                    fanins.append([pinname,portname])
                else:
                    pin_type = 'sink'
                    self.nets[netname].sink_pins.append(pinname)
                    cap = cell_info_map[cell_name]['pin_info'][portname]['capacitance']
                    if cap=='': cap = '13.06'
                    self.nets[netname].total_output_cap += float(cap)
                    if portname in ['CE','CE1','CE2']: fanins.append([pinname,portname])
                node = (pinname,
                        {'net': netname,
                         'cell_type':cell_name,
                         'port':portname,
                         'pin_type': pin_type,
                         'fanout':1,
                         'position':pin_location,
                         'DFF': 'DFF' in cell_name})
                sub_nodes.append(node)

        for fo_pinname,fo_portname in fanouts:
            #fo_pinname = '{}/{}'.format(instance_name, fo.portname)
            for fi_pinname,fi_portname in fanins:
                # for registers, we only add the edge between clock pin and output pin
                if cell_lib[cell_name]['pin_info'][fo_portname]['timing_tabs'].get(fi_portname,None) is None:
                    #print(cell_name,fo_portname)
                    continue
                edge = (fi_pinname,fo_pinname,{'edge_type':'cell', 'cell_type':cell_name})
                sub_edges.append(edge)

        return sub_nodes,sub_edges

    def parse_cell(self,cell_name,instance_name,portlist,io2arg,call_path):

        """

        given an instance of a cell, parse the cell into a sub-graph,
        where the input/output ports (that are connected to outside)
        are replaced with nets from father module.
                e.g., Test t1 (.A(a), .B(b));

        :param cell_name: str
                    the cell name
                        e.g., Test
        :param portlist: PortList
                    list of the instance's ports
        :param io2arg: dict{str:(str,int)}
                    {input/output net name: (port argument name, trace depth)}
                    dictionary that maps the input/output ports to arguments in father/grandpa module
                        trace_depth=1: father ; trace_back=2: grandpa
        :param call_path: str
                    the calling path of the module
                        e.g., ChipTop/.../
        :return:
            nodes : List[Tuple[str, Dict[str, str]]
                        list of the nodes, where each node correspond to a pin
                        Dict gives the feature of the pin
            edges: List[Tuple[str, str, Dict[str, bool]]
                        list of the edges, where each edge corresponds to a cell-edge/net edge
                        Dict gives the feature of the cell/net

        """

        # position = self.pin_loc_map[full_instance_name]

        idx = re.search('(x|xp|x\d+p)\d+',cell_name)
        cell_type = cell_name[:idx.start()]
        if cell_type.startswith('CK'):
            cell_type = cell_type[2:]
        self.cell_type_count[cell_type] = self.cell_type_count.get(cell_type,0)
        self.cell_type_count[cell_type] += 1
        sub_nodes = []
        sub_edges = []
        fanins, fanouts = [],[]
        # parse the ports (pins) of the cell one by one
        port_info_list = []
        for port in portlist:
            port_info = self.parse_cellport(cell_name,port)
            # distinguish fanin / fanout
            port_info_list.append(port_info)
            #if port_info.ptype == 'fanin':
            if port_info.ptype in ["CLK", "fanin"]:
                fanins.append(port_info)
            elif port_info.ptype == "fanout":
                fanouts.append(port_info)

        # if find a cell with no fanout, then print out
        if not fanouts:
            print(cell_name)
            print("***** warning, the above gate has no fanout recognized! *****")
            # do not assert, because some gates indeed have no fanout...
            # assert False, "no fanout recognized"

        # we treat each pin (the fanin/fanout of the cell) as a node
        if call_path != '':
            instance_name = '{}/{}'.format(call_path, instance_name)
        instance_name = instance_name.replace('\\', '')

        for port in port_info_list:
            port_netname = port.arg_list[0]
            # we may need to update the net name if it is an io of the module
            port_netname = update_netname(port_netname, call_path, io2arg)
            port_netname = port_netname.replace('\\', '')
            port_pinname = '{}/{}'.format(instance_name, port.portname)

            # get the location of the current cell instance
            if self.pin_loc_map.get(port_pinname,None) is None:
                assert False, 'pin with no location: {}'.format(port_pinname)
            pin_location = self.pin_loc_map[port_pinname]
            #if port.portname == 'GCLK':
            #    print(cell_name,port_pinname,port.ptype)
            # link the pin with the net
            if self.nets.get(port_netname, None) is None:
                self.nets[port_netname] = NetInfo(port_netname)
            if port.ptype == 'fanout':
                pin_type = 'drive'
                self.nets[port_netname].drive_pin = port_pinname
                self.nets[port_netname].drive_cell = cell_name
            elif port.ptype == 'fanin':
                pin_type = 'sink'
                self.nets[port_netname].sink_pins.append(port_pinname)
                cap = float(cell_info_map[cell_name]['pin_info'][port.portname]['capacitance'])
                self.nets[port_netname].total_output_cap += cap
            elif port.ptype == 'CLK':
                pin_type = 'sink'
                self.nets[port_netname].sink_pins.append(port_pinname)
                cap =float(cell_info_map[cell_name]['pin_info'][port.portname]['capacitance'])
                self.nets[port_netname].total_output_cap += cap
            else:
                assert  False, 'Wrong port type for {} {} {}'.format(cell_name,port.portname,port.ptype)
            node = (port_pinname,
                    {'net': port_netname,
                     'cell_type':cell_name,
                     'port':port.portname,
                     'pin_type': pin_type,
                     'fanout':1,
                     'position':pin_location,
                     'DFF':'DFF' in cell_name})
            sub_nodes.append(node)
        is_register = self.is_register(cell_name)
        # we do not add edges between input/output pin of registers!
        #if not is_register:
        #if not 'DFF' in cell_name and not 'SDFH' in cell_name:
            #print(cell_name)
        for fo in fanouts:
            fo_pinname = '{}/{}'.format(instance_name, fo.portname)
            for fi in fanins:
                #if 'ICG' in cell_name:
                #    print(instance_name,fi.portname)
                if is_register and 'clk' not in fi.portname.lower():
                    #print(cell_name, fi.portname)
                    continue
                fi_pinname = '{}/{}'.format(instance_name, fi.portname)
                sub_edges.append(
                    (fi_pinname,
                    fo_pinname,
                    {'edge_type':'cell','cell_type':cell_name})
                )

        return sub_nodes,sub_edges


    def is_register(self,cell_name):
        type = cell_info_map[cell_name]['type']
        return type in ("ASYNC_DFFH","DFFHQN","DFFHQ","DFFLQN",
                        "DFFLQ","DHL","DLL","ICG","SDFH", "SDFL")

    def parse_moduleport(
            self, port,father_wires
    ) -> PortInfo:
        """

        parse the information about a port of a cell

        :param port: Port
        :param father_wires: dict{str:(str, int,int)}
                    {wire_name: (wire_type, high bit, low bit)}
                    dictionary of wires in the father module
                    used to update the net names connected to father module

        :return:
            port_info: PortInfo
                the information of the port
        """

        # parse the argument
        portname, argname = port.portname, port.argname
        port_info = PortInfo(portname)
        if type(argname) == pyverilog.vparser.ast.Concat:
            args = argname.children()
            for arg in args:
                parse_arg(arg, port_info, father_wires)
        else:
            parse_arg(argname, port_info, father_wires)

        return port_info

    def parse_cellport(
        self, cell,port
    ) -> PortInfo:
        r"""

        parse the information about a port of a cell

        :param port: Port

        :return:
            port_info: PortInfo
                the information of the port
        """

        # find parse the port name and argument name
        portname, argname = port.portname, port.argname
        if type(argname) == pyverilog.vparser.ast.Partselect:
            arg_nm, high_bit, low_bit = argname.children()
            # print('\t parselct cell port arg:',portname, arg_nm, high_bit, low_bit)
        # argument is a pointer e.g., a[1]
        if type(argname) == pyverilog.vparser.ast.Pointer:
            argname = '{}[{}]'.format(str(argname.var),str(argname.ptr))
        # argument is a constant, e.g., 1'b0
        elif type(argname) == pyverilog.vparser.ast.IntConst:
            argname = argname.__str__()
        # argument is a identifier, e.g., b
        else:
            argname = argname.__str__()
        port_info = PortInfo(portname)
        port_info.arg_list = [argname]

        # find the type of the port: clock / fanin / fanout
        if self.is_output_port(cell,portname):
            port_info.ptype = "fanout"
        elif 'clk' in portname.lower():  # clock
            port_info.ptype = "CLK"
        #elif self.is_output_port(cell,portname):
        #    port_info.ptype = "fanout"
        else:
            port_info.ptype = "fanin"

        return port_info


    def check_path(self,graph,path):
        """

        given a path, check whether it exists in the given graph

        :param graph: networkx.Digraph
        :param path: List[str]
            list of nodes that form the path

        :return:
            flag: boolean
                True if path found, else False
        """

        flag = True
        pre_nd = path[0]
        stop_point = None
        for nd in path[1:]:
            if not graph.has_edge(pre_nd, nd):
                flag = False
                stop_point = nd
                break
            pre_nd = nd

        return flag,stop_point

    def parse_netlist(self, fname):
        r"""

        parse the netlist

        :param fname: str
            netlist filepath
        :return:
            nodes : List[Tuple[str, Dict[str, str]]
                        list of the nodes, where each node correspond to a pin
                        Dict gives the feature of the pin
            edges: List[Tuple[str, str, Dict[str, bool]]
                        list of the edges, where each edge corresponds to a cell-edge/net edge
                        Dict gives the feature of the cell/net
        """

        # load/build the abstract syntax tree that represents the netlist
        ast_save_path = os.path.join(self.data_path,'ast.pkl')
        if not os.path.exists(ast_save_path):
            #print(fname)
            ast, directives = parse([fname])
            with open(ast_save_path,'wb') as f:
                pickle.dump(ast,f)
        else:
            with open(ast_save_path,'rb') as f:
                ast = pickle.load(f)

        #ast, directives = parse([fname])
        self.ast = ast
        # extract the name of the modules
        start_time = time()
        modulelist = []
        for module in ast.description.definitions:
            modulelist.append(module.name)
        self.modulelist = modulelist

        # search for the top module
        top_module_name = self.top_module
        top_module = None
        print('\tsearching for the top module...')
        for module in ast.description.definitions:
            if module.name == top_module_name:
                top_module = module
                break
        assert top_module is not None, "top module {} not found".format(self.top_module)

        # parse the netlist
        nodes, edges = self.parse_module(top_module_name,
                                         None,None,'')
        num_cell_egdes = len(edges)
        # net = self.nets['system/tile_prci_domain/tile_reset_domain/tile/sha3_io_ptw_0_resp_bits_pte_ppn[19]_danc_841']
        # print(net.drive_pin,net.sink_pins)
        # exit()
        # add the net-edges

        # add the equavalent nets
        self.equal_net_map = {}
        for net in self.equal_wire_map.keys():
            equal_net = self.equal_wire_map[net]
            while self.equal_wire_map.get(equal_net,None) is not None:
                equal_net = self.equal_wire_map[equal_net]
            self.equal_net_map[net] = equal_net
        for net1,net2 in self.equal_net_map.items():
            if self.nets.get(net2) is None:
                # nodes.append(
                #     (net2,
                #     {'net':net2,'type':'PI','DFF':True,
                #     'position':self.pin_loc_map['{}/{}'.format(net2,net2)]})
                # )
                # nodes.append(
                #     (net1,
                #     {'net':net1,'type':'PI','DFF':True,
                #     'position':self.pin_loc_map['{}/{}'.format(net1,net1)]})
                # )
                continue
            else:
                drive_cell, net_name,drive_pin,sink_pins = self.nets[net2].drive_cell,self.nets[net2].net_name, \
                                                        self.nets[net2].drive_pin, self.nets[net2].sink_pins,
                total_output_cap = self.nets[net2].total_output_cap

                self.nets[net1] = NetInfo(net_name)
                self.nets[net1].drive_cell = drive_cell
                self.nets[net1].drive_pin = drive_pin
                self.nets[net1].sink_pins = sink_pins
                self.nets[net1].total_output_cap = total_output_cap

        pin2outcap = {}
        print('--Coonecting nets...')
        PIs = set()
        for net, net_info in self.nets.items():
            drive_pin = net_info.drive_pin
            # deal with the PIs: those nets with no drive pin
            #     we take the netname as the drive pin name
            #     also note that we should add a new node to the graph
            if drive_pin == '':
                net_info.drive_pin = net
                drive_pin = net
                position = self.pin_loc_map.get('{}/{}'.format(net,net),None)
                if position is None:
                    position = self.pin_loc_map[net_info.sink_pins[0]]
                nodes.append(
                    (net,
                     {'net':net,'cell_type':'PI','DFF':True,'position':position})
                )
                bin_x,bin_y = position[2:]
                PIs.add(net)
            else:
                bin_x, bin_y = self.pin_loc_map[drive_pin][2:]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bin_x,bin_y,bin_x,bin_y

            pin2outcap[drive_pin] = net_info.total_output_cap

            sink_pins = net_info.sink_pins
            num_sinks = len(sink_pins)
            #print('net: {}, \n\tdrive_pin:{}'.format(net,net_info.drive_pin),
            # '\n\tsink pins:',net_info.sink_pins)

            for sink_pin in sink_pins:
                edges.append(
                    (drive_pin,
                    sink_pin,
                    {'edge_type':'net','net_name':net})
                )

                # calculate the bounding box
                if '/' not in sink_pin:
                    sink_pin = '{}/{}'.format(sink_pin,sink_pin)
                bin_x,bin_y = self.pin_loc_map[sink_pin][2:]
                bbox_x1 = min(bbox_x1,bin_x)
                bbox_y1 = min(bbox_y1,bin_y)
                bbox_x2 = max(bbox_x2, bin_x)
                bbox_y2 = max(bbox_y2, bin_y)
            self.net_bbox_map[net] = [bbox_x1,bbox_y1,bbox_x2,bbox_y2]
            #print('\tnet :{},'.format(net),'bbox: ',self.net_bbox_map[net])
        # build the graph
        total_num_edges = len(edges)
        num_net_edges = total_num_edges - num_cell_egdes
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        self.graph = graph
        # new_cell_type_count = {}
        # for i in sorted(self.cell_type_count):
        #     new_cell_type_count[i] = self.cell_type_count[i]
        # print(new_cell_type_count)
        # path_count = {}
        # for path_info in self.timing_paths:
        #     len1 = len(path_info.path)
        #     path_count[len1] = path_count.get(len1,0)
        #     path_count[len1] += 1
        # new_path_count ={}
        # for i in sorted(path_count):
        #     new_path_count[i] = path_count[i]
        # print(new_path_count)
        # exit()
        end = time()
        print('--- Graph successfully built! num nodes: {}, num_edges: {}, spent time: {}'
              .format(graph.number_of_nodes(), graph.number_of_edges(),end-start_time))
        print('\t num cell-egdes: {}, num net-edges: {}\n'.format(num_cell_egdes,num_net_edges))

        #nds = list(self.graph.successors('clk'))
        #print(nds)
        #for nd in nds:
        #    print(nd,list(self.graph.successors(nd)))
        #exit()
        start2=time()
        print('--- Calculating the topological levels')
        # PIs are the startpoints of the timing paths
        #PIs = set()
        POs = set()
        PO2pathID = {}
        PO_in_level = {}
        num_po = 0
        for i,path_info in enumerate(self.timing_paths):

            #PIs.add(path_info.start)
            POs.add(path_info.end)
            PO2pathID[path_info.end] = i
        topo_levels = self.cal_topo_level(PIs,POs,PO2pathID)
        self.topo_levels = topo_levels
        self.node2level = {}
        for i,(nodes,_,_) in enumerate(topo_levels):
            for node in nodes:
                self.node2level[node] = i
        ept2path = {}
        #for ept in self.endpoints:
        for i,path_info in enumerate(self.timing_paths):
            ept = path_info.end
            report_path = path_info.path
            our_path = self.find_critical_path(ept)
            #print(len(report_path),len(our_path),len(set(report_path)-set(our_path)))
            ept2path[ept] = our_path


        total_count = 0
        end2 = time()
        # print(len(PIs),'system/tile_prci_domain/tile_reset_domain/tile/frontend/icache/s2_dout_3_reg[14]/SE' in PIs)
        print('\t num topological level: {}, spent time: {}'.format(len(topo_levels),end2-start2))
        all_sucs = []
        for i,(nodes,targets,path_ids) in enumerate(topo_levels):
            print('\t\tlevel {}\t#nodes: {},\t#targets: {}'.format(i,len(nodes),len(targets)))
        print('\t num remained nodes: {}, num remained edges: {}'
              .format(self.graph.number_of_nodes(), self.graph.number_of_edges()))
        print(total_count)
        print(len(all_sucs))

        #exit()
        if not os.path.exists('node_edges.pkl') :
            with open('node_edges.pkl','wb') as f:
                pickle.dump((nodes,edges),f)



        # check whether the timing paths exist in the graph
        print('--- Checking the existance of timing paths...')
        no_path = []
        for i, path_info in enumerate(self.timing_paths):
            # print('path {},'.format(i+1),'critcal:',True if path_info.is_critical else False)
            # print('\t  start: ',path_info.start)
            # print('\t  end: ',path_info.end)
            # if some nodes along the path is  not found, then flag=False, and stop_point gives the unfound node
            # else, flag = True
            flag,stop_point = self.check_path(self.graph,path_info.path)
            if not flag:
                no_path.append((i, path_info.start,path_info.end,stop_point))

        # check if all the timing paths are found
        if len(no_path) != 0:
            print('The following some timing paths can not be found in the netlist!')
            for i, start, end, stop in no_path:
                print('\tpath {}: start {}, end {}, stopped at {}'.format(i, start, end, stop))
            assert False
        print('--- Check passed!')

        # generate the region mask for each path, stored in the form of sparse matrix
        self.path_masks = th.zeros((self.num_paths,map_size_x,map_size_y))
        print('--- generating mask for each path...')
        indices = [
                [],
                []
            ]
        values = []
        for i,path_info in enumerate(self.timing_paths):
            #idx_map= {}
            idxs = []
            if self.masking == 'critical':
                #path = path_info.path
                path = ept2path[path_info.end]
                for j in range(len(path)-1):
                    #drive = path[i]
                    drive_loc = self.pin_loc_map.get(path[j],None)
                    if drive_loc is None:
                        drive_loc = self.pin_loc_map['{}/{}'.format(path[j],path[j])][2:]
                    else:
                        drive_loc = self.pin_loc_map[path[j]][2:]
                    sink_loc = self.pin_loc_map.get(path[j+1],None)
                    if sink_loc is None:
                        sink_loc = self.pin_loc_map['{}/{}'.format(path[j+1],path[j+1])][2:]
                    else:
                        sink_loc = self.pin_loc_map[path[j+1]][2:]
                    # the mask involves the boudning box area of each drive-sink net
                    x1 = min(drive_loc[0],sink_loc[0])
                    y1 = min(drive_loc[1],sink_loc[1])
                    x2 = max (drive_loc[0],sink_loc[0])
                    y2 = max(drive_loc[1],sink_loc[1])
                    for x in range(x1, x2 + 1):
                        idxs.extend(range(x * map_size_y + y1, x * map_size_y + y2 + 1))
                        #for v in range(x * map_size_y + y1, x * map_size_y + y2 + 1):
                        #    idx_map[v] = idx_map.get(v,0) + 1
                        #idxs.extend(range(x * map_size_y + y1, x * map_size_y + y2 + 1))

            elif self.masking == 'sibling':
                print('sibling')
                exit()
                nets = path_info.nets
                for net in nets:
                    bbox = self.net_bbox_map[net]
                    x1, y1, x2, y2 = bbox
                    for x in range(x1, x2 + 1):
                        #for v in range(x * map_size_y + y1, x * map_size_y + y2 + 1):
                        #    idx_map[v] = idx_map.get(v,0) + 1
                        idxs.extend(range(x * map_size_y + y1, x * map_size_y + y2 + 1))
            else:
                assert False, 'Wrong masking technique: {}, It shoule be in [critical, sibling]!'.format(self.masking)
            idxs = list(set(idxs))
            #idxs = list(idx_map.keys())
            indices[0].extend([i]*len(idxs))
            indices[1].extend(idxs)
            values.extend([1]*len(idxs))
            #values.extend(list(idx_map.values()))
            # self.path_masks = self.path_masks.reshape((self.num_paths,-1,1))
            # #path_info.mask = path_info.mask.reshape((-1,1))
            # self.path_masks[i][idxs] = 1
            # mask_1 = self.path_masks[i]
            # #mask_2 = th.sparse_coo_tensor(indices,values,(map_size_x,map_size_y))
            # #assert mask_1.equal(mask_2.to_device())
            # self.path_masks = self.path_masks.reshape((self.num_paths,map_size_x,map_size_y))
            #if len(path_info.path)<=10: print('\tpath {},end:{}, len:{}, arrival_time:{}, mask_len: {}, {}'.format(i,path_info.end,len(path_info.path),path_info.arrival_time,len(idxs),round(len(idxs)/(map_size_x*map_size_y),3) ))
            # if i==1:
            #     print(self.path_masks[i][30])

        mask_2 = th.sparse_coo_tensor(indices,values,(self.num_paths,map_size_x*map_size_y))
        self.path_masks = mask_2
        return self.graph,topo_levels,self.path_masks, PIs,pin2outcap

    def parse(self,data_dir):
        R"""

        parse a given design

        :param netlist_dir: str
            directory of the netlist of the design
        :param preopt_report_dir: str
            directory of the pre-optimization timing report
        :param postopt_report_dir: str
            directory of the post-optimization timing report
        :return:
            nodes : List[Tuple[str, Dict[str, str]]
                        list of the nodes, where each node correspond to a pin
                        Dict gives the feature of the pin
            edges: List[Tuple[str, str, Dict[str, bool]]
                        list of the edges, where each edge corresponds to a cell-edge/net edge
                        Dict gives the feature of the cell/net
        """
        self.data_path = data_dir
        netlist_path = os.path.join(data_dir, 'post-place/post-place.v')  # the netlist
        preopt_report_path = os.path.join(data_dir, 'post-place/path.tarpt')  # pre-optimization timing analysis report
        #new_postopt_report_path = os.path.join(data_dir, 'w_opt/new_path.tarpt')
        postopt_report_path = os.path.join(data_dir,
                                          'post-route/path.tarpt')  # post-optimization timing analysis report
        pre_pin_loc_file = os.path.join(data_dir, 'positions/pin_bin.txt')  # a file that contains the location of each instance
        #post_pin_loc_file = os.path.join(data_dir, 'positions/post_pinbin.txt')
        # bin_size_file = os.path.join(data_dir, 'positions/bin_size.txt')

        # with open(bin_size_file, 'r') as f:
        #     lines = f.readlines()
        # self.bin_size_x = float(lines[0].replace('\n', ''))
        # self.bin_size_y = float(lines[1].replace('\n', ''))

        start_time = time()
        print('--- Start parsing the netlist...')
        # you must call parse_postoptReport before parse_preoptReport to get the critical endpoints!
        self.parse_postoptReport(postopt_report_path)
        #self.parse_postoptReport(new_postopt_report_path)
        #return nx.DiGraph(),[],[],th.zeros((1,map_size_x,map_size_y))
        # cell_loc = parse_cell_locations(cell_loc_file)
        # self.pin_loc_map = cell_loc
        pre_pin_loc = self.parse_pin_locations(pre_pin_loc_file)
        save_path = os.path.join(self.data_path,'pre_pin2loc.pkl')
        with open(save_path,'wb') as f:
            pickle.dump(pre_pin_loc,f)
        # post_pin_loc = self.parse_pin_locations(post_pin_loc_file)
        # save_path = os.path.join(self.data_path,'post_pin2loc.pkl')
        # with open(save_path,'wb') as f:
        #     pickle.dump(post_pin_loc,f)
        self.pin_loc_map = pre_pin_loc

        timing_paths = self.parse_preoptReport(preopt_report_path)

        graph,topo_levels,path_masks,PIs,pin2outcap = self.parse_netlist(netlist_path)
        print('--- Parsing is done!')
        end_time = time()
        print('cost time: ', end_time - start_time)
        print('\n\n')
        return graph,topo_levels,timing_paths,path_masks,PIs,pin2outcap, self.pin2delay,self.pin2trans

    def find_critical_path(self,endpoint):
        cur_node = endpoint
        cur_level = self.node2level[cur_node]
        path = [endpoint]
        flag = False
        while cur_level>=2:
            pre_nodes = self.graph.predecessors(cur_node)
            for nd in pre_nodes:
                if 'clk' in nd.lower():
                    flag = True
                    break
                if self.node2level[nd] == cur_level-1:
                    path.append(nd)
                    cur_level -= 1
                    cur_node = nd
                    break
            if flag: break
        return path

    def cal_topo_level(self, PIs:set,POs:set,PO2pathID:dict):

        """

        calculate the topological levels of the graph,
        and remove the nodes that are not included in any level

        :param PIs: set(str)
            the points to begin the calculation

        :return:
            topo_levels: List[List[str]]
                list of topological levels, where each level contains a list of nodes

        """

        topo_levels = []
        remaining_nodes = PIs.copy()
        cur_nodes = list(PIs)
        topo_levels.append(cur_nodes)

        # continiously add the succussors of current level to form the next level, and then move to next level
        # until there is no node in current level
        while True:
            sucessors = []
            ready_nodes = []
            # add the succussors of current level to the next level
            for nd in cur_nodes:
                sucessors.extend(self.graph.successors(nd))
            sucessors = set(sucessors)
            cur_nodes = list(sucessors)

            # when there is no node left, break
            if len(sucessors)==0:
                break
            # add the current level and target nodes list of current level to the result
            topo_levels.append(cur_nodes)
            # add nodes in current level to remaining nodes
            remaining_nodes = remaining_nodes.union(cur_nodes)

        # reverse vist the topo levels, and remove the visited nodes in each level
        # this is to guarantee that each node only appear once
        visited = set()
        reverse_levels = []
        topo_levels.reverse()
        pre_nodes = []
        for rlevel in topo_levels:
            new_rlevel = set(rlevel)-visited
            visited = visited.union(new_rlevel)
            new_rlevel = list(new_rlevel)
            targets = list(POs.intersection(new_rlevel))
            path_ids = [PO2pathID[po] for po in targets]
            # we also save the information of target_node list in each level
            reverse_levels.append(
                (new_rlevel,
                 targets,
                 path_ids)
            )
        reverse_levels.reverse()
        topo_levels = reverse_levels

        # remove all the nodes that do not belong to any level
        removed_nodes = set(self.graph.nodes()) - remaining_nodes
        self.graph.remove_nodes_from(list(removed_nodes))

        return topo_levels

def main():
    postopt_report_path = '../data/preCTSOptTiming_new/ChipTop_preCTS_all.tarpt'
    preopt_report_path = '../data/preCTS_new/preCTS_all.tarpt'
    #report_path = '../data/ChipTop_preCTS_all.tarpt'
    #netlist_path = '../data/ChipTop.mapped(3).v'
    netlist_path = '../data/netlist_after_placement.v'
    pin_loc_file = '../rawdata'
    parser = Parser(top_module='ChipTop',masking='critical')
    print(os.listdir('chacha'))
    parser.data_path = 'data'
    parser.parse_postoptReport('chacha/post-route/path.tarpt')
    parser.parse_preoptReport('chacha/post-place/path.tarpt')


if __name__ == "__main__":
    seed = 1234
    main()

