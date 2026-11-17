import { installTooltips } from '@window-shell/browser/tooltip';
import {
  Box, ChartLine, ChevronDown, Circle, FileText, FlaskConical, Layers, Maximize, Plus, Search, Settings, Waypoints,
  ZoomIn, ZoomOut,
} from 'lucide-react';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { setAppearance } from '@/appearance/appearance';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { Button, IconButton } from '@/components/ui/Button';
import { Card, CardHead } from '@/components/ui/Card';
import { Checkbox } from '@/components/ui/Checkbox';
import { Chip } from '@/components/ui/Chip';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { Combobox } from '@/components/ui/Combobox';
import { Dialog } from '@/components/ui/Dialog';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { ListRow } from '@/components/ui/ListRow';
import {
  ContextMenu, ContextMenuContent, ContextMenuTrigger, Menu, MenuContent, MenuGroup, MenuGroupLabel, MenuItem,
  MenuSeparator, MenuTrigger,
} from '@/components/ui/Menu';
import { NumberField } from '@/components/ui/NumberField';
import { PanelHeader } from '@/components/ui/PanelHeader';
import { ScrollFade } from '@/components/ui/ScrollFade';
import { Section } from '@/components/ui/Section';
import { Popover, PopoverClose, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Slider } from '@/components/ui/Slider';
import { Select, type Option } from '@/components/ui/Select';
import { StatusMark, type StatusMarkStatus } from '@/components/ui/StatusMark';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { TextField } from '@/components/ui/TextField';
import { ValueFieldGallery } from './ValueFieldGallery';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';

type Theme = 'light' | 'dark';
type Density = 'compact' | 'regular';
type Labels = 'beside' | 'above';
type ValueMark = 'hover' | 'quiet';
type CardStyle = 'quiet' | 'ring';

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
  return (
    <div className="grid gap-2">
      <div className="h-72 w-72 overflow-auto rounded-control border border-shell-divider">
        <Tree ref={tree} aria-label="Outline" nodes={treeNodes(outline, names)} defaultExpanded={['model']}
          selected={selected} onSelectedChange={setSelected} onRename={rename}
          menu={(node) => (
            <>
              <MenuItem onClick={() => tree.current?.rename(node.id)}>Rename</MenuItem>
              <MenuItem>Duplicate</MenuItem>
              <MenuItem>Remove</MenuItem>
            </>
          )} />
      </div>
      <div className="flex gap-2">
        <Button onClick={() => { setSelected(['noise']); }}>Select Noise</Button>
        <Button onClick={() => tree.current?.reveal('muscles')}>Reveal Muscles</Button>
      </div>
      <span className="text-xs text-shell-muted" data-testid="tree-selection">{selected.join(', ')}</span>
    </div>
  );
}

const components = ['Linear', 'GRU', 'LSTM', 'MLP', 'Noise', 'Delay', 'Arm', 'Muscles', 'Point Mass', 'Feedback'];

function DrawerDemo() {
  const [shown, setShown] = useState(true);
  const [open, setOpen] = useState<Record<string, boolean>>({ library: true, outline: true });
  const [selected, setSelected] = useState<string[]>(['gru']);
  const fold = (section: string) => ({
    open: open[section], onOpenChange: (next: boolean) => setOpen((now) => ({ ...now, [section]: next })),
  });
  return (
    <div className="grid gap-2">
      <div className="flex h-96 w-64 flex-col rounded-control border border-shell-divider" data-testid="drawer">
        {shown ? <>
          <Section name="Library" count={components.length} {...fold('library')}
            actions={<IconButton size="mark" label="New Component" icon={<Plus />} />}>
            <div className="grid gap-px p-1">
              {components.map((name) => <ListRow key={name} tabIndex={0} name={name} glyph={<Box />} />)}
            </div>
          </Section>
          <Section name="Outline" {...fold('outline')}>
            <Tree aria-label="Outline" nodes={treeNodes(outline, {})} defaultExpanded={['model', 'encoder']}
              selected={selected} onSelectedChange={setSelected} />
          </Section>
          <Section name="Exceptions" count={0} defaultOpen={false}>
            <div className="p-2 text-sm text-shell-muted">None</div>
          </Section>
        </> : null}
      </div>
      <Button onClick={() => setShown(!shown)}>{shown ? 'Hide' : 'Show'}</Button>
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
              {sweep.map((value) => <Chip key={value}>{value}</Chip>)}
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

export function Gallery() {
  useLayoutEffect(() => installTooltips(), []);
  const theme = useRootAttribute<Theme>('data-theme');
  const density = useRootAttribute<Density>('data-density');
  const labels = useRootAttribute<Labels>('data-inspector-labels');
  const valueMark = useRootAttribute<ValueMark>('data-value-mark');
  const [cardStyle, setCardStyle] = useState<CardStyle>('quiet');
  useLayoutEffect(() => {
    if (cardStyle === 'quiet') delete document.documentElement.dataset.card;
    else document.documentElement.dataset.card = cardStyle;
  }, [cardStyle]);
  const [text, setText] = useState('policy');
  const [number, setNumber] = useState<number | null>(0.001);
  const [position, setPosition] = useState(0.6);
  const [checked, setChecked] = useState(true);
  const [kind, setKind] = useState<string | null>('train');
  const [optimizer, setOptimizer] = useState<string | null>('adam');
  const [method, setMethod] = useState('feedbax.supervised_gradient');
  const [windowState, setWindowState] = useState('both');
  const [library, setLibrary] = useState('library');
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
        <SegmentedControl<Labels> aria-label="Inspector labels" value={labels}
          onValueChange={(value) => setAppearance(window, 'inspector-labels', value)}
          items={[{ value: 'beside', label: 'Beside' }, { value: 'above', label: 'Above' }]} />
        <SegmentedControl<ValueMark> aria-label="Value mark" value={valueMark}
          onValueChange={(value) => setAppearance(window, 'value-mark', value)}
          items={[{ value: 'hover', label: 'Hover' }, { value: 'quiet', label: 'Quiet' }]} />
        <SegmentedControl<CardStyle> aria-label="Card" value={cardStyle} onValueChange={setCardStyle}
          items={[{ value: 'quiet', label: 'Quiet' }, { value: 'ring', label: 'Ring' }]} />
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
        </Group>

        <Group name="Combobox">
          <State name="value"><Combobox aria-label="Optimizer" items={optimizers} value={optimizer} onValueChange={setOptimizer} className="w-48" /></State>
          <State name="placeholder"><Combobox aria-label="Optimizer" items={optimizers} value={null} placeholder="Optimizer" onValueChange={() => undefined} className="w-48" /></State>
          <State name="invalid"><Combobox aria-label="Optimizer" items={optimizers} value={null} invalid onValueChange={() => undefined} className="w-48" /></State>
          <State name="disabled"><Combobox aria-label="Optimizer" items={optimizers} value="sgd" disabled onValueChange={() => undefined} className="w-48" /></State>
        </Group>

        <Group name="CatalogPicker">
          <State name="value"><CatalogPicker aria-label="Training method" className="w-64" value={method} onChoose={setMethod}
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

        <Group name="Tree">
          <TreeDemo />
        </Group>

        <Group name="Section">
          <DrawerDemo />
        </Group>

        <Group name="ScrollFade">
          <ScrollFadeDemo />
        </Group>

        <Group name="Card">
          <CardDemo />
        </Group>

        <Group name="InlineEdit">
          <State name={editingTitle ? 'editing' : 'closed'}>
            <div className="flex h-row w-48 items-center rounded-row px-1 hover:bg-shell-hover" tabIndex={0}
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

        <Group name="Chip">
          <State name="document"><Chip glyph={<Box />}>policy</Chip></State>
          <State name="output"><Chip glyph={<Circle />} trailing={<StatusMark status="succeeded" />}>Train · checkpoint</Chip></State>
          <State name="selected"><Chip glyph={<Box />} selected>policy</Chip></State>
          <State name="long"><Chip glyph={<Box />} className="w-32">a very long document name</Chip></State>
        </Group>

        <Group name="Table">
          <div className="w-full max-w-xl overflow-hidden rounded-control border border-shell-divider">
            <Table>
              <thead><tr><TableHeaderCell>Run</TableHeaderCell><TableHeaderCell>Seed</TableHeaderCell><TableHeaderCell>Loss</TableHeaderCell></tr></thead>
              <tbody>
                <TableRow tabIndex={0}><TableCell>run-1</TableCell><TableCell>1</TableCell><TableCell className="tabular-nums">0.412</TableCell></TableRow>
                <TableRow tabIndex={0} selected><TableCell>run-2</TableCell><TableCell>2</TableCell><TableCell className="tabular-nums">0.388</TableCell></TableRow>
                <TableRow tabIndex={0} related><TableCell>run-3</TableCell><TableCell>2</TableCell><TableCell className="tabular-nums">0.401</TableCell></TableRow>
              </tbody>
            </Table>
          </div>
        </Group>

        <Group name="PanelHeader">
          <div className="w-80 rounded-control border border-shell-divider">
            <PanelHeader title="Details" glyph={<FileText />}
              actions={<><IconButton label="Fit" icon={<Maximize />} /><IconButton label="Settings" icon={<Settings />} /></>} />
          </div>
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
