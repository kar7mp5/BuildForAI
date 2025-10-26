import pygame
import time
import sys

# Logic gate
def not_gate(a):
    return 1 if a == 0 else 0

def and_gate(a, b):
    return 1 if a == 1 and b == 1 else 0

def or_gate(a, b):
    return 1 if a == 1 or b == 1 else 0

def xor_gate(a, b):
    return 1 if a != b else 0

def nor_gate(a, b):
    return not_gate(or_gate(a, b))

# 1-bit ALU
def alu_1bit(a, b, cin, opcode, prev_b=None, next_b=None, bit_index=0):
    op0, op1, op2 = opcode[0], opcode[1], opcode[2]
    
    b_sub = xor_gate(b, op0)
    sum_xor1 = xor_gate(a, b_sub)
    result_add = xor_gate(sum_xor1, cin)
    carry1 = and_gate(a, b_sub)
    carry2 = and_gate(sum_xor1, cin)
    cout_add_sub = or_gate(carry1, carry2)
    
    result_sub = result_add
    result_and = and_gate(a, b)
    result_or = or_gate(a, b)
    result_xor = xor_gate(a, b)
    
    if op0 == 1 and op1 == 0 and op2 == 1:  # 0b101
        result_shl = prev_b if bit_index > 0 else cin
        cout_shl = b
    else:
        result_shl = 0
        cout_shl = cout_add_sub
    
    if op0 == 0 and op1 == 1 and op2 == 1:  # 0b110
        result_shr = next_b if bit_index < 7 else cin
        cout_shr = b if bit_index == 0 else cin
    else:
        result_shr = 0
        cout_shr = cout_add_sub
    
    sel_add = and_gate(and_gate(not_gate(op2), not_gate(op1)), not_gate(op0))
    sel_sub = and_gate(and_gate(not_gate(op2), not_gate(op1)), op0)
    sel_and = and_gate(and_gate(not_gate(op2), op1), not_gate(op0))
    sel_or = and_gate(and_gate(not_gate(op2), op1), op0)
    sel_xor = and_gate(and_gate(op2, not_gate(op1)), not_gate(op0))
    sel_shl = and_gate(and_gate(op2, not_gate(op1)), op0)
    sel_shr = and_gate(and_gate(op2, op1), not_gate(op0))
    
    result = or_gate(or_gate(or_gate(or_gate(or_gate(or_gate(
        and_gate(result_add, sel_add),
        and_gate(result_sub, sel_sub)),
        and_gate(result_and, sel_and)),
        and_gate(result_or, sel_or)),
        and_gate(result_xor, sel_xor)),
        and_gate(result_shl, sel_shl)),
        and_gate(result_shr, sel_shr))
    
    cout = cout_shl if sel_shl == 1 else (cout_shr if sel_shr == 1 else cout_add_sub)
    
    return result, cout

# 8-bit ALU
def alu_8bit(A, B, opcode):
    result = []
    cout = 0
    for i in range(8):
        prev_b = B[i-1] if i > 0 else 0
        next_b = B[i+1] if i < 7 else 0
        cin = opcode[0] if i == 0 else cout  # Fixed: cin = op0 for first bit (1 for SUB, 0 for ADD)
        res, cout = alu_1bit(A[i], B[i], cin, opcode, prev_b, next_b, i)
        result.append(res)
    zero = 0 if any(result) else 1
    return result, cout, zero

# Simplified 8-bit D Flip-Flop
def d_flip_flop_8bit_update(D_bits, clock, states):
    Q_bits = []
    Q_bar_bits = []
    for i in range(8):
        prev_clock = states[i].get('prev_clock', 0)
        Q = states[i].get('Q', 0)
        rising_edge = clock == 1 and prev_clock == 0
        if rising_edge:
            states[i]['Q'] = D_bits[i]
        states[i]['prev_clock'] = clock
        Q_bits.append(states[i]['Q'])
        Q_bar_bits.append(not_gate(states[i]['Q']))
    return Q_bits, Q_bar_bits, states

# 4-bit D Flip-Flop
def d_flip_flop_4bit_update(D_bits, clock, states):
    Q_bits = []
    Q_bar_bits = []
    for i in range(4):
        prev_clock = states[i].get('prev_clock', 0)
        Q = states[i].get('Q', 0)
        rising_edge = clock == 1 and prev_clock == 0
        if rising_edge:
            states[i]['Q'] = D_bits[i]
        states[i]['prev_clock'] = clock
        Q_bits.append(states[i]['Q'])
        Q_bar_bits.append(not_gate(states[i]['Q']))
    return Q_bits, Q_bar_bits, states

# Helper: Bits to int
def bits_to_int(bits):
    return sum(bit * (2**i) for i, bit in enumerate(bits))

# Helper: Int to 8-bit list (LSB first)
def int_to_bits_8(value):
    if value < 0 or value > 255:
        raise ValueError("Value must be 0-255")
    bits = []
    for i in range(8):
        bits.append((value >> i) & 1)
    return bits

# Helper: Int to 4-bit list (LSB first)
def int_to_bits_4(value):
    if value < 0 or value > 15:
        raise ValueError("Value must be 0-15")
    bits = []
    for i in range(4):
        bits.append((value >> i) & 1)
    return bits

# Helper: Opcode to operation name
def opcode_to_name(opcode):
    op_map = {
        (0, 0, 0, 0): "NOP",
        (1, 0, 0, 0): "LDA",
        (0, 1, 0, 0): "STA",
        (1, 1, 0, 0): "ADD",
        (0, 0, 1, 0): "SUB",
        (1, 0, 1, 0): "OUT",
        (0, 1, 1, 0): "JMP",
        (1, 1, 1, 0): "HLT"
    }
    return op_map.get(tuple(opcode), "UNKNOWN")

# 7-segment display patterns (0-9)
seven_segment = {
    0: [1, 1, 1, 1, 1, 1, 0],  # 0
    1: [0, 1, 1, 0, 0, 0, 0],  # 1
    2: [1, 1, 0, 1, 1, 0, 1],  # 2
    3: [1, 1, 1, 1, 0, 0, 1],  # 3
    4: [0, 1, 1, 0, 0, 1, 1],  # 4
    5: [1, 0, 1, 1, 0, 1, 1],  # 5
    6: [1, 0, 1, 1, 1, 1, 1],  # 6
    7: [1, 1, 1, 0, 0, 0, 0],  # 7
    8: [1, 1, 1, 1, 1, 1, 1],  # 8
    9: [1, 1, 1, 1, 0, 1, 1]   # 9
}

def draw_7_segment(screen, value, x, y, scale=1.0, color_on=(0, 255, 0), color_off=(20, 20, 20)):
    segments = seven_segment.get(value % 10, [0]*7)
    seg_width, seg_height = int(20 * scale), int(5 * scale)
    positions = [
        ((x, y - 20*scale), (x + seg_width, y - 20*scale)),  # Top
        ((x + seg_width, y - 20*scale), (x + seg_width, y)),  # Top-right
        ((x + seg_width, y), (x + seg_width, y + 20*scale)),  # Bottom-right
        ((x, y + 20*scale), (x + seg_width, y + 20*scale)),  # Bottom
        ((x, y), (x, y + 20*scale)),  # Bottom-left
        ((x, y - 20*scale), (x, y)),  # Top-left
        ((x, y), (x + seg_width, y))  # Middle
    ]
    for i, on in enumerate(segments):
        color = color_on if on else color_off
        pygame.draw.line(screen, color, positions[i][0], positions[i][1], seg_height)

# Parse user input program
def parse_program(input_str):
    op_map = {
        "NOP": 0b0000,
        "LDA": 0b0001,
        "STA": 0b0010,
        "ADD": 0b0011,
        "SUB": 0b0100,
        "OUT": 0b0101,
        "JMP": 0b0110,
        "HLT": 0b0111
    }
    ram = [0] * 256
    instructions = input_str.split(';')
    addr = 0
    for instr in instructions:
        instr = instr.strip()
        if not instr:
            continue
        parts = instr.split()
        op = parts[0].upper()
        if op not in op_map:
            raise ValueError(f"Unknown instruction: {op}")
        opcode = op_map[op]
        if len(parts) > 1:
            arg = int(parts[1])
            if arg < 0 or arg > 15:
                raise ValueError(f"Address/data {arg} out of range (0-15)")
            ram[addr] = (opcode << 4) | arg
        else:
            ram[addr] = opcode << 4
        addr += 1
        if addr >= 16:
            break
    return ram

# CPU class to manage registers, memory, and control unit
class CPU:
    def __init__(self, program):
        self.ram = parse_program(program)
        self.ram[10] = 5  # Preload data
        self.ram[11] = 3  # Preload data
        self.a_reg = [0] * 8
        self.b_reg = [0] * 8
        self.pc = [0] * 8
        self.ir = [0] * 8
        self.mar = [0] * 8
        self.out_reg = [0] * 8
        self.bus = [0] * 8
        self.zero = 0
        self.carry = 0
        self.halted = False
        self.a_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]
        self.b_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]
        self.out_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]
        self.pc_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]
        self.mar_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]
        self.ir_states = [{'Q': 0, 'prev_clock': 0} for _ in range(8)]

    def fetch(self, sim_clock):
        self.mar, _, self.mar_states = d_flip_flop_8bit_update(self.pc, sim_clock, self.mar_states)
        addr = bits_to_int(self.mar)
        if 0 <= addr < len(self.ram):
            instr = self.ram[addr]
            self.bus = int_to_bits_8(instr)
        else:
            self.bus = [0] * 8
        self.ir, _, self.ir_states = d_flip_flop_8bit_update(self.bus, sim_clock, self.ir_states)
        pc_val = bits_to_int(self.pc)
        new_pc = int_to_bits_8((pc_val + 1) % 256)
        self.pc, _, self.pc_states = d_flip_flop_8bit_update(new_pc, sim_clock, self.pc_states)

    def decode_execute(self, sim_clock, cycle_count):
        if self.halted:
            return
        opcode = self.ir[4:]
        operand = bits_to_int(self.ir[0:4])
        op_name = opcode_to_name(opcode)
        print(f"Cycle {cycle_count}: PC={bits_to_int(self.pc)}, IR={bits_to_int(self.ir):02X}, Opcode={op_name}, Operand={operand}, A={bits_to_int(self.a_reg)}, Out={bits_to_int(self.out_reg)}")

        if op_name == "NOP":
            pass
        elif op_name == "LDA":
            self.mar, _, self.mar_states = d_flip_flop_8bit_update(int_to_bits_8(operand), sim_clock, self.mar_states)
            addr = bits_to_int(self.mar)
            if 0 <= addr < len(self.ram):
                self.bus = int_to_bits_8(self.ram[addr])
            self.a_reg, _, self.a_states = d_flip_flop_8bit_update(self.bus, sim_clock, self.a_states)
        elif op_name == "STA":
            self.mar, _, self.mar_states = d_flip_flop_8bit_update(int_to_bits_8(operand), sim_clock, self.mar_states)
            addr = bits_to_int(self.mar)
            if 0 <= addr < len(self.ram):
                self.bus = self.a_reg[:]
                self.we = 1
                self.ram[addr] = bits_to_int(self.bus) & 0xFF
        elif op_name == "ADD" or op_name == "SUB":
            self.mar, _, self.mar_states = d_flip_flop_8bit_update(int_to_bits_8(operand), sim_clock, self.mar_states)
            addr = bits_to_int(self.mar)
            if 0 <= addr < len(self.ram):
                self.bus = int_to_bits_8(self.ram[addr])
                self.b_reg, _, self.b_states = d_flip_flop_8bit_update(self.bus, sim_clock, self.b_states)
                alu_opcode = [0, 0, 0] if op_name == "ADD" else [1, 0, 0]
                result, self.carry, self.zero = alu_8bit(self.a_reg, self.b_reg, alu_opcode)
                self.a_reg, _, self.a_states = d_flip_flop_8bit_update(result, sim_clock, self.a_states)
        elif op_name == "OUT":
            self.bus = self.a_reg[:]
            self.out_reg, _, self.out_states = d_flip_flop_8bit_update(self.bus, sim_clock, self.out_states)
        elif op_name == "JMP":
            new_pc = int_to_bits_8(operand)
            self.pc, _, self.pc_states = d_flip_flop_8bit_update(new_pc, sim_clock, self.pc_states)
        elif op_name == "HLT":
            self.halted = True

# Real-time simulation with Pygame
def real_time_computer_simulation(program, cycle_duration=1.0):
    pygame.init()
    screen_width = 1000
    screen_height = 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("8-Bit Computer Simulation")
    clock = pygame.time.Clock()

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

    font = pygame.font.SysFont('Arial', 18)

    led_radius = 12
    led_spacing = 30
    row_spacing = 40
    start_x = 150
    start_y = 50

    labels = ["Clock", "PC [7:0]", "MAR [7:0]", "IR [7:0]", "A [7:0]", "B [7:0]", "Bus [7:0]", "Out [7:0]", "Carry", "Zero"]
    led_counts = [1, 8, 8, 8, 8, 8, 8, 8, 1, 1]

    cpu = CPU(program)
    cycle_count = 0
    sim_clock = 0
    cycle_start_ticks = pygame.time.get_ticks()
    paused = False
    current_cycle_duration = cycle_duration * 1000
    fetch_phase = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    cpu = CPU(program)
                    cycle_count = 0
                    sim_clock = 0
                    cycle_start_ticks = pygame.time.get_ticks()
                    paused = False
                    fetch_phase = True
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    current_cycle_duration += 500
                elif event.key == pygame.K_MINUS:
                    current_cycle_duration = max(100, current_cycle_duration - 500)

        if not paused and not cpu.halted:
            current_ticks = pygame.time.get_ticks()
            if current_ticks - cycle_start_ticks >= current_cycle_duration / 2:
                sim_clock = 1 - sim_clock
                cycle_start_ticks = current_ticks

                if sim_clock == 0:
                    for reg_states in [cpu.a_states, cpu.b_states, cpu.out_states, cpu.pc_states, cpu.mar_states, cpu.ir_states]:
                        for bit_state in reg_states:
                            bit_state['prev_clock'] = 0

                if sim_clock == 1:
                    cycle_count += 1
                    if fetch_phase:
                        cpu.fetch(sim_clock)
                        fetch_phase = False
                    else:
                        cpu.decode_execute(sim_clock, cycle_count)
                        fetch_phase = True

        screen.fill(BLACK)

        for row, (label, count) in enumerate(zip(labels, led_counts)):
            label_text = font.render(label, True, WHITE)
            screen.blit(label_text, (20, start_y + row * row_spacing))

            for i in range(count):
                x = start_x + i * led_spacing
                y = start_y + row * row_spacing
                state = 0
                if label == "Clock":
                    state = sim_clock
                elif label == "PC [7:0]":
                    state = cpu.pc[7 - i] if i < 8 else 0
                elif label == "MAR [7:0]":
                    state = cpu.mar[7 - i] if i < 8 else 0
                elif label == "IR [7:0]":
                    state = cpu.ir[7 - i]
                elif label == "A [7:0]":
                    state = cpu.a_reg[7 - i]
                elif label == "B [7:0]":
                    state = cpu.b_reg[7 - i]
                elif label == "Bus [7:0]":
                    state = cpu.bus[7 - i]
                elif label == "Out [7:0]":
                    state = cpu.out_reg[7 - i]
                elif label == "Carry":
                    state = cpu.carry
                elif label == "Zero":
                    state = cpu.zero

                color = GREEN if state == 1 else RED
                pygame.draw.circle(screen, color, (x, y), led_radius)

        out_decimal = bits_to_int(cpu.out_reg)
        draw_7_segment(screen, out_decimal // 100, 700, 500, scale=1.5)
        draw_7_segment(screen, (out_decimal // 10) % 10, 750, 500, scale=1.5)
        draw_7_segment(screen, out_decimal % 10, 800, 500, scale=1.5)

        ram_text = "RAM: " + " ".join(f"{i:01X}:{cpu.ram[i]:02X}" for i in range(16))
        ram_display = font.render(ram_text, True, WHITE)
        screen.blit(ram_display, (20, screen_height - 100))

        op_name = opcode_to_name(cpu.ir[4:])
        info_text = font.render(f"Cycle: {cycle_count} | Instr: {op_name} | PC: {bits_to_int(cpu.pc)} | Out: {out_decimal} | Speed: {current_cycle_duration / 1000:.1f}s/cycle", True, WHITE)
        screen.blit(info_text, (20, screen_height - 40))

        if paused:
            status_text = font.render("Paused (Press 'p' to resume)", True, YELLOW)
            screen.blit(status_text, (20, screen_height - 70))
        elif cpu.halted:
            status_text = font.render("Halted (Press 'r' to reset)", True, YELLOW)
            screen.blit(status_text, (20, screen_height - 70))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    program = "LDA 10; ADD 11; ADD 11; ADD 11; ADD 11; ADD 11; ADD 11; OUT; HLT"
    real_time_computer_simulation(program, cycle_duration=1.0)
    