import { installTooltips } from '@window-shell/browser/tooltip';
import {
  Box, ChartLine, ChevronDown, Circle, Cpu, FileText, FlaskConical, Grid2x2, Layers, Maximize, Play, Plus, Search, Settings, Square,
  SquareArrowOutUpRight, StepForward, Waypoints, ZoomIn, ZoomOut, type LucideIcon,
} from 'lucide-react';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { setAppearance, type AppearanceValue } from '@/appearance/appearance';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { Button, IconButton } from '@/components/ui/Button';
import { Card, CardHead } from '@/components/ui/Card';
import { CommandBar, type BarCommand, type BarEntry } from '@/components/ui/CommandBar';
import { Checkbox } from '@/components/ui/Checkbox';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { Combobox } from '@/components/ui/Combobox';
import { Dialog } from '@/components/ui/Dialog';
import { HoverCard, HoverCardTrigger, createHoverCardHandle } from '@/components/ui/HoverCard';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { ListRow } from '@/components/ui/ListRow';
import {
  ContextMenu, ContextMenuContent, ContextMenuTrigger, Menu, MenuCheckboxItem, MenuContent, MenuGroup, MenuGroupLabel, MenuItem,
  MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuTrigger,
} from '@/components/ui/Menu';
import { NumberField } from '@/components/ui/NumberField';
import { ScrollFade } from '@/components/ui/ScrollFade';
import { Section, SectionStack } from '@/components/ui/Section';
import { Popover, PopoverClose, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Slider } from '@/components/ui/Slider';
import { Select, type Option } from '@/components/ui/Select';
import { StatusMark, type StatusMarkStatus } from '@/components/ui/StatusMark';
import { Table, TableCell, TableFieldCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import { TextField } from '@/components/ui/TextField';
import { ValueFieldGallery } from './ValueFieldGallery';
import { RunGridGallery } from './RunGridGallery';
import { RuleFieldGallery } from './RuleFieldGallery';
import { CardStyles } from './CardStyles';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';

type Theme = 'light' | 'dark';
type Density = 'compact' | 'regular';
type CardStyle = AppearanceValue<'card'>;

const kinds = [
  { value: 'train', label: 'Train', glyph: <Box /> },
  { value: 'evaluate', label: 'Evaluate', glyph: <FlaskConical /> },
  { value: 'analyze', label: 'Analyze', glyph: <ChartLine /> },
  { value: 'report', label: 'Report', glyph: <FileText />, disabled: true },
] as const satisfies readonly Option<string>[];

const optimizers: Option<string>[] = [
  'adam', 'adamw', 'adabelief', 'adagrad', 'lamb', 'lion', 'nadam', 'radam', 'rmsprop', 'sgd',
].map((value) => ({ value, label: value }));

const statuses: [StatusMarkStatus, string][] = [
  ['succeeded', 'Done'], ['running', 'Running'], ['queued', 'Queued'], ['failed', 'Failed'], ['cannot_run', "Can't run"],
  ['stale', 'Out of date'], ['unchecked', 'Unchecked'], ['checking', 'Checking'], ['environment', 'Environment'],
  ['not_run', 'Not run'], ['starting', 'Starting'], ['stopping', 'Stopping'], ['collecting', 'Downloading'],
  ['awaiting_authorization', 'Needs approval'], ['blocked', 'Blocked'], ['stopped', 'Stopped'],
  ['interrupted', 'Interrupted'], ['corrupt', 'Damaged'],
];

type Outline = { id: string; name: string; kind: 'composite' | 'component' | 'attachment'; status?: StatusMarkStatus; children?: Outline[] };

const outline: Outline[] = [
  { id: 'model', name: 'Model', kind: 'composite', children: [
    { id: 'encoder', name: 'Encoder', kind: 'composite', children: [
      { id: 'embed', name: 'Embed', kind: 'component' },
      { id: 'gru', name: 'GRU', kind: 'component', status: 'stale', children: [
        { id: 'noise', name: 'Noise', kind: 'attachment' },
      ] },
      { id: 'readout', name: 'Readout', kind: 'component' },
    ] },
    { id: 'mechanics', name: 'Mechanics', kind: 'composite', children: [
      { id: 'arm', name: 'Arm', kind: 'component', status: 'failed' },
      { id: 'muscles', name: 'Muscles', kind: 'component' },
    ] },
    { id: 'feedback', name: 'Feedback', kind: 'component' },
  ] },
  { id: 'objective', name: 'Objective', kind: 'component', status: 'succeeded' },
];

const kindGlyphs = { composite: <Layers />, component: <Box />, attachment: <Waypoints /> };

function treeNodes(items: readonly Outline[], names: Readonly<Record<string, string>>): TreeNode[] {
  return items.map((item) => ({
    id: item.id,
    name: names[item.id] ?? item.name,
    glyph: kindGlyphs[item.kind],
    trailing: item.status ? <StatusMark status={item.status} /> : undefined,
    children: item.children && treeNodes(item.children, names),
  }));
}

function TreeDemo() {
  const tree = useRef<TreeHandle>(null);
  const [selected, setSelected] = useState<string[]>(['gru']);
  const [names, setNames] = useState<Record<string, string>>({});
  const rename = (id: string, name: string) => setNames((current) => ({ ...current, [id]: name }));
  const [entered, setEntered] = useState<string>();
  return (
    <div className="grid gap-2">
      <ScrollFade axis="both" className="h-72 w-72 rounded-control border border-shell-divider">
        <Tree ref={tree} aria-label="Outline" nodes={treeNodes(outline, names)} defaultExpanded={['model']}
          selected={selected} onSelectedChange={setSelected} onRename={rename} onActivate={setEntered}
          menu={(node) => (
            <>
              <MenuItem onClick={() => tree.current?.rename(node.id)}>Rename</MenuItem>
              <MenuItem>Duplicate</MenuItem>
              <MenuItem>Remove</MenuItem>
            </>
          )} />
      </ScrollFade>
      <div className="flex gap-2">
        <Button onClick={() => { setSelected(['noise']); }}>Select Noise</Button>
        <Button onClick={() => tree.current?.reveal('muscles')}>Reveal Muscles</Button>
      </div>
      <span className="text-xs text-shell-muted" data-testid="tree-selection">{selected.join(', ')}</span>
      <span className="text-xs text-shell-muted" data-testid="tree-entered">{entered}</span>
    </div>
  );
}

const components = ['Linear', 'GRU', 'LSTM', 'MLP', 'Noise', 'Delay', 'Arm', 'Muscles', 'Point Mass', 'Feedback'];

const libraryGroups = [
  { name: 'Networks', items: ['Linear', 'GRU', 'LSTM', 'MLP'] },
  { name: 'Mechanics', items: ['Arm', 'Muscles', 'Point Mass', 'Delay', 'Noise', 'Feedback'] },
  { name: 'Runtime', items: Array.from({ length: 24 }, (_, index) => `Operator ${index + 1}`) },
];

function DrawerDemo({ content }: { content: 'short' | 'overflowing' }) {
  const [shown, setShown] = useState(true);
  const [open, setOpen] = useState<Record<string, boolean>>({ library: true, outline: true, Runtime: true });
  const [selected, setSelected] = useState<string[]>(['gru']);
  const fold = (section: string) => ({
    open: open[section] ?? false, onOpenChange: (next: boolean) => setOpen((now) => ({ ...now, [section]: next })),
  });
  const groups = content === 'short' ? libraryGroups.slice(0, 1) : libraryGroups;
  const [card] = useState(() => createHoverCardHandle<string>());
  return (
    <State name={content}>
      <div className="h-96 w-64 rounded-control border border-shell-divider bg-shell-sidebar" data-testid={`drawer-${content}`}>
        {shown ? <SectionStack>
          <Section name="Library" {...fold('library')}
            actions={<IconButton size="mark" label="New Component" icon={<Plus />} />}
            lead={<div className="border-b border-[color:var(--neutral-alpha-06)] p-2">
              <TextField aria-label="Search Library" leading={<Search />} />
            </div>}>
            {groups.map((group) => <Section key={group.name} level="group" name={group.name} {...fold(group.name)}>
              <div className="grid gap-px px-1 pb-1">
                {group.items.map((name) => <HoverCardTrigger key={name} handle={card} payload={name}
                  render={<ListRow tabIndex={0} name={name} glyph={<Box />} draggable />} />)}
              </div>
            </Section>)}
          </Section>
          <HoverCard handle={card} beside={(trigger) => trigger.closest('[data-section-stack]')}>
            {(name) => <div className="grid gap-2">
              <div className="font-medium">{name}</div>
              <p className="text-shell-muted">A component that transforms its input.</p>
            </div>}
          </HoverCard>
          <Section name="Outline" {...fold('outline')}>
            <Tree aria-label="Outline" nodes={treeNodes(content === 'short' ? outline.slice(0, 1) : outline, {})}
              defaultExpanded={['model', 'encoder']} selected={selected} onSelectedChange={setSelected} />
          </Section>
        </SectionStack> : null}
      </div>
      <Button onClick={() => setShown(!shown)}>{shown ? 'Hide' : 'Show'}</Button>
    </State>
  );
}

function FieldGroupDemo() {
  return (
    <div className="w-64 rounded-control border border-shell-divider p-1">
      <Section level="field" name="Optimizer">
        <div className="grid gap-2 px-2 pb-2">
          <TextField aria-label="Learning Rate" defaultValue="0.001" />
          <TextField aria-label="Beta" defaultValue="0.9" />
        </div>
      </Section>
    </div>
  );
}

const sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

function ScrollFadeDemo() {
  return (
    <>
      <State name="horizontal">
        <div className="w-72 rounded-control border border-shell-divider">
          <ScrollFade axis="x" data-testid="fade-x">
            <div className="flex w-max gap-1 p-1">
              {sweep.map((value) => <Button key={value} variant="ghost" size="mark">{value}</Button>)}
            </div>
          </ScrollFade>
        </div>
      </State>
      <State name="vertical">
        <div className="flex h-40 w-48 flex-col rounded-control border border-shell-divider">
          <ScrollFade data-testid="fade-y">
            <div className="grid gap-px p-1">
              {components.map((name) => <ListRow key={name} tabIndex={0} name={name} glyph={<Box />} />)}
            </div>
          </ScrollFade>
        </div>
      </State>
    </>
  );
}

function CardDemo() {
  const [selected, setSelected] = useState('train');
  const steps = [['train', 'train', <Box key="g" />], ['evaluate', 'evaluate', <FlaskConical key="g" />]] as const;
  return (
    <>
      {steps.map(([id, name, icon]) => (
        <State key={id} name={selected === id ? 'selected' : 'default'}>
          <Card selected={selected === id} className="w-52">
            <CardHead>
              <Button variant="bare" size="content" align="start" onClick={() => setSelected(id)}
                className="flex h-row w-full items-center gap-1.5 rounded-row px-2 [&_svg]:text-shell-muted">
                {icon}{name}
              </Button>
            </CardHead>
            <div className="flex gap-1 px-2 pb-2 pt-1">
              {[0, 1, 2, 3].map((run) => <StatusMark key={run} status={run === 3 ? 'running' : 'succeeded'} />)}
            </div>
          </Card>
        </State>
      ))}
    </>
  );
}

function Group({ name, children }: { name: string; children: ReactNode }) {
  return (
    <section className="grid gap-2" aria-label={name}>
      <h2 className="text-sm font-medium text-shell-muted">{name}</h2>
      <div className="flex flex-wrap items-center gap-3">{children}</div>
    </section>
  );
}

function State({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="grid justify-items-start gap-1">
      {children}
      <span className="text-xs text-shell-muted">{name}</span>
    </div>
  );
}

const runs = [['run-1', 1, 0.412], ['run-2', 2, 0.388], ['run-3', 2, 0.401]] as const;

/** The same table in a box narrower than its least width, which scrolls on its own, and in one wide enough. */
function TableDemo() {
  const [rates, setRates] = useState<Record<string, number>>({ 'run-1': 0.001, 'run-2': 0.003, 'run-3': 0.01 });
  const [names, setNames] = useState<string[]>(runs.map(([name]) => name));
  const [renaming, setRenaming] = useState<number>();
  const table = (label: string) => <Table aria-label={label} held={1}
    columns={[{ least: '7rem', fixed: true }, { least: '5rem' }, { least: '7rem' }, { least: '6rem' }]}
    addRow={{ label: 'Add Run', onAdd: () => setNames([...names, `run-${names.length + 1}`]) }}
    box={{ className: 'rounded-control border border-shell-divider' }}>
    <thead><tr>
      <TableHeaderCell scope="col">Run</TableHeaderCell><TableHeaderCell scope="col">Seed</TableHeaderCell>
      <TableHeaderCell scope="col">Learning rate</TableHeaderCell><TableHeaderCell scope="col">Loss</TableHeaderCell>
    </tr></thead>
    <tbody>{names.map((_, index) => [`run-${index + 1}`, ...runs[index % runs.length].slice(1)] as const).map(([run, seed, loss], index) =>
      <TableRow key={run} tabIndex={0} selected={index === 1} related={index === 2}>
      <TableFieldCell header>
        <InlineEdit form="cell" aria-label={`Rename ${names[index]}`} value={names[index]} editing={renaming === index}
          onEditingChange={(editing) => setRenaming(editing ? index : undefined)}
          onCommit={(name) => setNames(names.map((old, at) => at === index ? name : old))} />
      </TableFieldCell>
      <TableCell>{seed}</TableCell>
      <TableFieldCell><ValueField form="cell" model={rawField({
        id: run, label: `Learning rate ${run}`, type: { kind: 'number' }, value: rates[run] ?? 0.001,
        change: (value) => setRates({ ...rates, [run]: value as number }),
      })} /></TableFieldCell>
      <TableCell className="tabular-nums">{loss}</TableCell>
    </TableRow>)}</tbody>
  </Table>;
  return <div className="grid w-full gap-3">
    <div className="w-64" data-gallery-box="narrow">{table('Runs, narrow')}</div>
    <div className="w-full max-w-xl" data-gallery-box="wide">{table('Runs, wide')}</div>
  </div>;
}

function CommandBarDemo() {
  const [last, setLast] = useState('');
  const command = (id: string, title: string, glyph: LucideIcon, enabled = true) =>
    ({ id, title, label: title, glyph, enabled, invoke: () => setLast(title) });
  const pinned = { action: command('run', 'Run', Play), counterpart: command('stop', 'Stop', Square, false) };
  const entries: BarEntry<BarCommand>[] = [
    { command: command('add', 'Add', Plus), priority: 'normal' },
    { command: command('resume', 'Resume', StepForward), priority: 'normal' },
    { command: command('open', 'Open Run', SquareArrowOutUpRight), priority: 'low' },
    { command: command('pivot', 'Pivot', Grid2x2), priority: 'low' },
  ];
  return (
    <div className="space-y-2">
      <div className="w-[min(100%,24rem)] resize-x overflow-hidden rounded-surface border border-shell-divider p-1">
        <CommandBar pinned={pinned} entries={entries} moreLabel="More" aria-label="Commands" />
      </div>
      <div className="text-sm text-shell-muted">{last}</div>
    </div>
  );
}

export function Gallery() {
  useLayoutEffect(() => installTooltips(), []);
  const theme = useRootAttribute<Theme>('data-theme');
  const density = useRootAttribute<Density>('data-density');
  const cardStyle = useRootAttribute<CardStyle>('data-card');
  const [text, setText] = useState('policy');
  const [number, setNumber] = useState<number | null>(0.001);
  const [position, setPosition] = useState(0.6);
  const [checked, setChecked] = useState(true);
  const [kind, setKind] = useState<(typeof kinds)[number]['value'] | null>('train');
  const [optimizer, setOptimizer] = useState<string | null>('adam');
  const [method, setMethod] = useState('feedbax.supervised_gradient');
  const [windowState, setWindowState] = useState<'map' | 'details' | 'both'>('both');
  const [library, setLibrary] = useState<'library' | 'outline'>('library');
  const [expanded, setExpanded] = useState(true);
  const [dialog, setDialog] = useState(false);
  const [chosen, setChosen] = useState('');
  const [title, setTitle] = useState('model 2');
  const [editingTitle, setEditingTitle] = useState(false);

  return (
    <main className="min-h-screen bg-shell-surface text-shell-foreground">
      <header className="sticky top-0 z-10 flex h-panel-header items-center gap-3 border-b border-shell-divider bg-shell-chrome px-panel-header-inline">
        <h1 className="flex-1 font-medium">Controls</h1>
        <SegmentedControl<Theme> aria-label="Theme" value={theme} onValueChange={(value) => setAppearance(window, 'theme', value)}
          items={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
        <SegmentedControl<Density> aria-label="Density" value={density}
          onValueChange={(value) => setAppearance(window, 'density', value)}
          items={[{ value: 'compact', label: 'Compact' }, { value: 'regular', label: 'Regular' }]} />
        <Select<CardStyle> aria-label="Cards" value={cardStyle} onValueChange={(value) => setAppearance(window, 'card', value)}
          items={[{ value: 'flow', label: 'React Flow' }, { value: 'flow-ui', label: 'React Flow UI' }, { value: 'turbo', label: 'Turbo' },
            { value: 'quiet', label: 'Quiet' }, { value: 'ring', label: 'Ring' }]} />
      </header>
      <div className="grid max-w-5xl gap-6 p-6">
        <Group name="Button">
          <State name="default"><Button>Train</Button></State>
          <State name="primary"><Button variant="primary">Run</Button></State>
          <State name="ghost"><Button variant="ghost">Cancel</Button></State>
          <State name="glyph"><Button><Plus />Add</Button></State>
          <State name="disabled"><Button disabled>Train</Button></State>
          <State name="primary disabled"><Button variant="primary" disabled>Run</Button></State>
          <State name="pressed"><Button variant="ghost" pressed>Outline</Button></State>
          <State name="start"><Button variant="ghost" align="start" className="w-40">GRU</Button></State>
          <State name="mark"><Button size="mark">3</Button></State>
        </Group>

        <Group name="IconButton">
          <State name="ghost"><IconButton label="Fit" icon={<Maximize />} /></State>
          <State name="default"><IconButton variant="default" label="Settings" icon={<Settings />} /></State>
          <State name="pressed"><IconButton label="Zoom in" icon={<ZoomIn />} pressed /></State>
          <State name="disabled"><IconButton label="Zoom out" icon={<ZoomOut />} disabled /></State>
          <State name="mark"><IconButton size="mark" label="Add Port" icon={<Plus />} /></State>
        </Group>

        <Group name="TextField">
          <State name="default"><TextField aria-label="Name" value={text} onValueChange={setText} /></State>
          <State name="placeholder"><TextField aria-label="Search" placeholder="Search" leading={<Search />} /></State>
          <State name="invalid"><TextField aria-label="Invalid" defaultValue="2x" invalid /></State>
          <State name="read-only"><TextField aria-label="Read-only" value="checkpoint-12" readOnly /></State>
          <State name="disabled"><TextField aria-label="Disabled" value="policy" disabled /></State>
        </Group>

        <Group name="NumberField">
          <State name="default"><NumberField aria-label="Learning rate" value={number} onValueChange={setNumber} step={0.001} /></State>
          <State name="bounded"><NumberField aria-label="Batch size" value={64} onValueChange={() => undefined} min={1} max={64} /></State>
          <State name="invalid"><NumberField aria-label="Epochs" value={-1} onValueChange={() => undefined} invalid /></State>
          <State name="read-only"><NumberField aria-label="Seed" value={7} onValueChange={() => undefined} readOnly /></State>
          <State name="disabled"><NumberField aria-label="Steps" value={100} onValueChange={() => undefined} disabled /></State>
        </Group>

        <Group name="Slider">
          <State name="default"><Slider aria-label="Position" value={position} onValueChange={setPosition} min={0} max={1} step={0.01} className="w-48" /></State>
          <State name="disabled"><Slider aria-label="Locked" value={0.3} onValueChange={() => undefined} min={0} max={1} step={0.01} disabled className="w-48" /></State>
        </Group>
        <Group name="Checkbox">
          <State name="checked"><Checkbox label="Shuffle" checked={checked} onCheckedChange={setChecked} /></State>
          <State name="unchecked"><Checkbox label="Clip" checked={false} onCheckedChange={() => undefined} /></State>
          <State name="indeterminate"><Checkbox label="All" checked={false} indeterminate onCheckedChange={() => undefined} /></State>
          <State name="disabled"><Checkbox label="Resume" checked disabled onCheckedChange={() => undefined} /></State>
          <State name="unlabelled"><Checkbox aria-label="Select row" checked onCheckedChange={() => undefined} /></State>
        </Group>

        <Group name="Select">
          <State name="value"><Select aria-label="Kind" items={kinds} value={kind} onValueChange={setKind} className="w-40" /></State>
          <State name="placeholder"><Select aria-label="Kind" items={kinds} value={null} placeholder="Kind" onValueChange={() => undefined} className="w-40" /></State>
          <State name="disabled"><Select aria-label="Kind" items={kinds} value="train" disabled onValueChange={() => undefined} className="w-40" /></State>
          <State name="leading"><Select aria-label="Kind" items={kinds} value={kind} onValueChange={setKind} leading={<Cpu />} className="w-40" /></State>
        </Group>

        <Group name="Combobox">
          <State name="value"><Combobox aria-label="Optimizer" items={optimizers} value={optimizer} onValueChange={setOptimizer} className="w-48" /></State>
          <State name="placeholder"><Combobox aria-label="Optimizer" items={optimizers} value={null} placeholder="Optimizer" onValueChange={() => undefined} className="w-48" /></State>
          <State name="invalid"><Combobox aria-label="Optimizer" items={optimizers} value={null} invalid onValueChange={() => undefined} className="w-48" /></State>
          <State name="disabled"><Combobox aria-label="Optimizer" items={optimizers} value="sgd" disabled onValueChange={() => undefined} className="w-48" /></State>
        </Group>

        <Group name="CatalogPicker">
          <State name="value"><CatalogPicker aria-label="Method" className="w-64" value={method} onChoose={setMethod}
            entries={[
              { key: 'feedbax.supervised_gradient', label: 'Supervised gradient', package: 'feedbax' },
              { key: 'lab.evolution', label: 'Evolution strategy', package: 'lab-methods' },
            ]} /></State>
        </Group>

        <Group name="Menu">
          <State name={chosen || 'closed'}>
            <Menu>
              <MenuTrigger render={<Button />}><Plus />Step<ChevronDown /></MenuTrigger>
              <MenuContent>
                {kinds.map((item) => (
                  <MenuItem key={item.value} glyph={item.glyph} disabled={'disabled' in item}
                    onClick={() => setChosen(item.label)}>{item.label}</MenuItem>
                ))}
                <MenuSeparator />
                <MenuGroup>
                  <MenuGroupLabel>Map</MenuGroupLabel>
                  <MenuItem shortcut="⇧1" onClick={() => setChosen('Fit')}>Fit</MenuItem>
                </MenuGroup>
                <MenuSeparator />
                <MenuCheckboxItem checked={chosen !== 'Hide grid'} onCheckedChange={(shown) => setChosen(shown ? '' : 'Hide grid')}>
                  Grid
                </MenuCheckboxItem>
                <MenuSeparator />
                <MenuRadioGroup value={chosen === 'arm' ? 'arm' : 'point-mass'} onValueChange={(value: string) => setChosen(value)}>
                  <MenuRadioItem value="point-mass">point-mass</MenuRadioItem>
                  <MenuRadioItem value="arm">arm</MenuRadioItem>
                </MenuRadioGroup>
              </MenuContent>
            </Menu>
          </State>
          <State name="context">
            <ContextMenu>
              <ContextMenuTrigger className="flex h-[5em] w-[12em] items-center justify-center rounded-control border border-dashed border-shell-control text-shell-muted">
                Map
              </ContextMenuTrigger>
              <ContextMenuContent>
                {kinds.map((item) => (
                  <MenuItem key={item.value} glyph={item.glyph} disabled={'disabled' in item}
                    onClick={() => setChosen(item.label)}>{item.label}</MenuItem>
                ))}
              </ContextMenuContent>
            </ContextMenu>
          </State>
        </Group>

        <Group name="Popover">
          <Popover>
            <PopoverTrigger render={<Button />}>Sweep</PopoverTrigger>
            <PopoverContent aria-label="Sweep" className="grid w-64 gap-2">
              <NumberField aria-label="Trials" value={8} onValueChange={() => undefined} />
              <PopoverClose render={<Button variant="primary" />}>Apply</PopoverClose>
            </PopoverContent>
          </Popover>
        </Group>

        <Group name="SegmentedControl">
          <State name="text">
            <SegmentedControl aria-label="Panel" value={library} onValueChange={setLibrary}
              items={[{ value: 'library', label: 'Library' }, { value: 'outline', label: 'Outline' }]} />
          </State>
          <State name="glyph">
            <SegmentedControl aria-label="Window" value={windowState} onValueChange={setWindowState}
              items={[
                { value: 'map', label: 'Map', icon: <Box /> },
                { value: 'both', label: 'Both', icon: <ChartLine /> },
                { value: 'details', label: 'Details', icon: <FileText />, disabled: true },
              ]} />
          </State>
          <State name="disabled">
            <SegmentedControl aria-label="Disabled" value="library" onValueChange={() => undefined} disabled
              items={[{ value: 'library', label: 'Library' }, { value: 'outline', label: 'Outline' }]} />
          </State>
        </Group>

        <Group name="ListRow">
          <div className="w-72 rounded-control border border-shell-divider p-1" role="tree" aria-label="Steps">
            <ListRow role="treeitem" tabIndex={0} name="Pipeline" glyph={<Box />} expanded={expanded} onExpandedChange={setExpanded} trailing="3" />
            {expanded ? <>
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Train" glyph={<Box />} selected trailing="running" />
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Evaluate" glyph={<FlaskConical />} trailing="done" />
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Report" glyph={<FileText />} disabled />
            </> : null}
          </div>
        </Group>

        <ValueFieldGallery />

        <Group name="Rule field">
          <RuleFieldGallery />
        </Group>

        <Group name="Card styles">
          <CardStyles />
        </Group>

        <Group name="Tree">
          <TreeDemo />
        </Group>

        <Group name="Section">
          <DrawerDemo content="short" />
          <DrawerDemo content="overflowing" />
          <FieldGroupDemo />
        </Group>

        <Group name="ScrollFade">
          <ScrollFadeDemo />
        </Group>

        <Group name="CommandBar">
          <CommandBarDemo />
        </Group>

        <Group name="Card">
          <CardDemo />
        </Group>

        <Group name="Run grid">
          <RunGridGallery />
        </Group>

        <Group name="InlineEdit">
          <State name={editingTitle ? 'editing' : 'closed'}>
            <div className="flex h-row w-48 items-center rounded-row px-1 hover:bg-shell-row-hover" tabIndex={0}
              onKeyDown={(event) => { if (event.key === 'Enter') setEditingTitle(true); }}
              onDoubleClick={() => setEditingTitle(true)}>
              <InlineEdit aria-label="Name" value={title} editing={editingTitle} onEditingChange={setEditingTitle} onCommit={setTitle} />
            </div>
          </State>
        </Group>

        <Group name="StatusMark">
          <div className="grid grid-cols-3 gap-x-6 gap-y-1">
            {statuses.map(([status, label]) => (
              <span key={status} className="flex h-row items-center gap-2 text-sm">
                <StatusMark status={status} label={label} />{label}
              </span>
            ))}
          </div>
        </Group>


        <Group name="Table">
          <TableDemo />
        </Group>


        <Group name="Dialog">
          <Button onClick={() => setDialog(true)}>Rename</Button>
          {dialog ? (
            <Dialog ariaLabel="Rename" manageFocus onClose={() => setDialog(false)} panelClassName="grid w-72 gap-3">
              <TextField aria-label="Name" defaultValue="policy" data-dialog-autofocus />
              <div className="flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setDialog(false)}>Cancel</Button>
                <Button variant="primary" onClick={() => setDialog(false)}>Rename</Button>
              </div>
            </Dialog>
          ) : null}
        </Group>
      </div>
    </main>
  );
}
