from parse.Cfg import Cfg
from parse.BlockType import BlockType
from parse.BasicBlock import BasicBlock
from parse.entity import OrderDict,Triplet,Stack,Logger
import re
BLOCK_LENGTH = 100
DISPATCHER_LENGTH = 1
FALLBACK_LENGTH = 1

# 对cfg进行分析
# 1.对dispatcher要判断其长度
# 2.对fallback函数进行分析
# 3.对每个函数function分析
# 分析的机制存在一定的问题，原本是只要有call就保留，但是还要考虑callvalue 5楼固定的值
class CfgAnalysis:
    def __init__(self,cfg:Cfg):
        self.cfg = cfg
        self.basicblocks = cfg.basicBlocks
        self.fliter_fallback = False
        self.fliter_dispatcher = False
        self.fliter_function = False
        self._fliterset = set()

    # 分析函数
    def analyse(self):
        # 首先判断这个cfg是不是比较短
        # 如果比较短的话就不分析了，所有的块都会保留
        # if self.cfg.short:
        #     return
        # 块有一定的长度，可以进行下一步分析
        # 分析fallback
        # if self.cfg.fallback_num/self.cfg.length < 0.5:
        self.check_fallback()
        # 分析每个函数function
        # if self.cfg.dispatcher_num/self.cfg.length < 0.5:
        self.check_function()
        # if len(self._fliterset)/self.cfg.length <0.5:
        self.filter()

    # 分析fallback函数，判断其中是否包含call
    def check_fallback(self):
        for offset in self.cfg.fallbacks:
            block:BasicBlock = self.basicblocks.get(offset)
            # for suc in block.successors:
            #     if suc.type == BlockType.COMMON:
            #         return
            if block.has_caller() and not self.is_zero_fallback(block):
                # 进一步判断是否为5字常见fallback块
                return
        self.fliter_fallback = True

    # 函数分析，对每一个函数遍历，里面只要有call就全部保留，同时
    def check_function(self):
        self.fliter_dispatcher = True
        allset = set()
        retainset = set()
        for offset in self.cfg.functions:
            call = False
            queue = Stack()
            visited = set()
            queue.push(offset)
            while not queue.isEmpty():
                eoffset = queue.pop()
                visited.add(eoffset)
                block:BasicBlock = self.basicblocks.get(eoffset)
                if block.has_caller() and not self.is_zero_fallback(block):
                    call =True
                for i in block.successors:
                    if (i.offset not in visited) and (i.offset not in self.cfg.dispatchers):
                        queue.push(i.offset)
            if call:
                retainset.update(visited)
            allset.update(visited)
        self._fliterset = allset - retainset
        if len(self._fliterset)/self.cfg.length < 0.5:
            self.fliter_function = True

    def filter(self):
        if self.fliter_fallback:
            for offset in self.cfg.fallbacks:
                block:BasicBlock = self.basicblocks.get(offset)
                block.setretain(False)
        if self.fliter_dispatcher:
            for offset in self.cfg.dispatchers:
                block:BasicBlock = self.basicblocks.get(offset)
                block.setretain(False)
        if self.fliter_function:
            for offset in self._fliterset:
                block:BasicBlock = self.basicblocks.get(offset)
                block.setretain(False)
    
    def is_zero_fallback(self,block:BasicBlock):
        if block.length>=5 and block.getInstruction(-1).name == "JUMPI" and\
                    "PUSH" in block.getInstruction(-2).name and \
                    block.getInstruction(-3).name == "ISZERO" and \
                    block.getInstruction(-4).name == "CALLVALUE" and\
                    block.getInstruction(-5).name == "JUMPDEST":
            return True
        return False
    
    def extract_list(self):
        contract_list = []
        blocks = self.basicblocks
        removedig = re.compile(r'[0-9]+')
        for pair in blocks:
            block:BasicBlock = pair.value
            if block.retain:
                ins = block.instructions
                blocklist = []
                for i in ins:
                    instr = i.name
                    word = removedig.sub('',instr)
                    if word != "":
                        blocklist.append(word)
                if blocklist != []:
                    contract_list.append(blocklist)
        return contract_list

    def analyse_icws(self):
        self.check_fallback()
        self.fliter_dispatcher = True
        self.filter()

    # # 曾经的分析函数，淘汰了
    # def analysis(self,cfg:Cfg):
    #     self.cfg = cfg
    #     self.basicblocks = cfg.basicBlocks
    #     if self.cfg.length < BLOCK_LENGTH:
    #         return
    #     if self.cfg.fallback_num > FALLBACK_LENGTH:
    #         self.analyse_fallback()
    #     if self.cfg.loader is not None:
    #         self.analyse_loader()
    #     if self.cfg.dispatcher_num > DISPATCHER_LENGTH:
    #         self.analyse_function()
    #     self.fliter()
        
    # # 原来的fallback函数
    # def analyse_fallback(self):
    #     if self.cfg.fallback_num/self.cfg.length > 0.5:
    #         return
    #     for offset in self.cfg.fallbacks:
    #         block:BasicBlock = self.basicblocks.get(offset)
    #         for suc in block.successors:
    #             if suc.type == BlockType.COMMON:
    #                 return
    #         if block.has_caller():
    #             return
    #     self.fliter_fallback = True
            
    # def analyse_loader(self):
    #     loaderoffset = self.cfg.loader
    #     if loaderoffset is None:
    #         return
    #     block:BasicBlock = self.basicblocks.get(loaderoffset)
    #     ins = block.instructions
    #     lens = block.length
    #     if lens > 3:
    #         last = block.getInstruction(-1)
    #         eq = block.getInstruction(-3)
    #         if "JUMP" in last.name and "EQ" in eq.name:
    #             self.fliter_loader = True

    # def analyse_function(self):
    #     allset = set()
    #     retainset = set()
    #     for offset in self.cfg.dispatchers:
    #         call = False
    #         visited = []
    #         queue = Stack()
    #         queue.push(offset)
    #         while not queue.isEmpty():
    #             elem = queue.pop()
    #             block:BasicBlock = self.basicblocks.get(elem)
    #             visited.append(elem)
    #             if block.has_caller():
    #                 call = True
    #             for i in block.successors:
    #                 if (i.offset not in visited) and (i.offset not in self.cfg.dispatchers):
    #                     queue.push(i.offset)
    #         if call:
    #             retainset.update(visited)
    #         allset.update(visited)
    #     self._fliterset = allset - retainset
    #     if len(self._fliterset)/self.cfg.length < 0.3:
    #         self.fliter_function = True


        